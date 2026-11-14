"""Google Drive access for the autograder: parse the link, and find the saved model files
without downloading the weights.

Supported links (from drive_link.txt)
    https://drive.google.com/drive/folders/<id>     a folder (the model folder, the submission
                                                    folder, or a folder holding the zip)
    https://drive.google.com/file/d/<id>/view       one file, normally <mission>_submission.zip
    https://drive.google.com/open?id=<id>, .../uc?id=<id>

How files are read
    - Folders are listed through Drive's public embedded folder view (no API key needed).
    - Files are read with HTTP Range requests on Drive's download endpoint, so only the
      bytes needed are fetched: sizes, config.json, and the few-KB safetensors header.
    - A zip is opened in place: zipfile seeks through a range-reading file object, so only
      its directory and the small members are fetched.

Testing: set AUTOGRADER_DRIVE_MOCK=/path/to/mock.json to serve ids from local files
(see autograder/tests/make_examples.py); MockFetcher has the same interface as HttpFetcher.
"""
import html
import io
import json
import os
import re
import urllib.parse
import zipfile

LINK_RE = re.compile(r"https?://(?:drive|docs)\.google\.com/\S+|https?://drive\.usercontent\.google\.com/\S+")
ID_PATTERNS = [
    ("folder", re.compile(r"/folders/([\w-]{10,})")),
    ("file", re.compile(r"/file/d/([\w-]{10,})")),
    ("unknown", re.compile(r"[?&]id=([\w-]{10,})")),
]

TIMEOUT_S = 20
MAX_FOLDER_LISTINGS = 12  # folders listed while searching for the model
SEARCH_DEPTH = 3  # e.g. shared folder -> submission/ -> model/


class DriveError(Exception):
    """kind: 'denied' (not shared publicly), 'not_found', 'network', or 'unexpected'."""

    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind


def parse_link(text):
    """Return (kind, id, url) for the first Drive link in `text`, or None.

    kind is 'folder', 'file', or 'unknown' (an ?id= link, which can be either).
    """
    for match in LINK_RE.finditer(text):
        url = match.group(0).rstrip(").,;'\">")
        for kind, pattern in ID_PATTERNS:
            m = pattern.search(url)
            if m:
                return kind, m.group(1), url
    return None


# Fetchers: list a folder, get a file's name and size, read a byte range
class HttpFetcher:
    DOWNLOAD = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
    FOLDER = "https://drive.google.com/embeddedfolderview?id={}"
    ENTRY_RE = re.compile(r'<div class="flip-entry" id="entry-([\w-]+)".*?<a href="([^"]+)".*?'
                          r'<div class="flip-entry-title">(.*?)</div>', re.S)

    def __init__(self):
        import requests
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "Mozilla/5.0 (CSE151B autograder)"
        self.info = {}  # id -> (name, size)

    def _request(self, url, headers=None):
        import requests
        try:
            resp = self.session.get(url, headers=headers or {}, stream=True, timeout=TIMEOUT_S)
        except requests.RequestException as e:
            raise DriveError("network", f"could not reach Google Drive ({type(e).__name__}: {e})")
        host = urllib.parse.urlparse(resp.url).netloc
        if "accounts.google.com" in host or resp.status_code in (401, 403):
            resp.close()
            raise DriveError("denied", "Google Drive asked for a sign-in, so the link is not shared publicly")
        if resp.status_code == 404:
            resp.close()
            raise DriveError("not_found", "Google Drive says this file or folder does not exist")
        if resp.status_code == 429 or resp.status_code >= 500:
            resp.close()
            raise DriveError("network", f"Google Drive returned HTTP {resp.status_code} (busy or rate limited)")
        if resp.status_code not in (200, 206):
            resp.close()
            raise DriveError("unexpected", f"Google Drive returned HTTP {resp.status_code}")
        return resp

    @staticmethod
    def _html_problem(text):
        low = text.lower()
        if "servicelogin" in low or "accounts.google.com" in low and "sign in" in low:
            return DriveError("denied", "Google Drive asked for a sign-in, so the link is not shared publicly")
        if "quota exceeded" in low or "too many users" in low:
            return DriveError("network", "Google Drive's download quota for this file is exceeded; try again later")
        if "does not exist" in low or "not found" in low:
            return DriveError("not_found", "Google Drive says this file does not exist")
        return DriveError("unexpected", "Google Drive returned a web page instead of the file")

    def list_folder(self, folder_id):
        resp = self._request(self.FOLDER.format(folder_id))
        text = resp.text
        if "flip-entries" not in text:
            raise self._html_problem(text)
        entries = []
        for fid, href, title in self.ENTRY_RE.findall(text):
            entries.append({"id": fid, "name": html.unescape(title).strip(), "is_folder": "/folders/" in href})
        return entries

    def _read(self, file_id, start, end):
        """Return (data, name, total_size) for bytes start..end (inclusive) of a file."""
        resp = self._request(self.DOWNLOAD.format(file_id), {"Range": f"bytes={start}-{end}"})
        try:
            if "text/html" in resp.headers.get("Content-Type", ""):
                raise self._html_problem(resp.text[:200_000])
            name = _filename(resp.headers.get("Content-Disposition", ""))
            if resp.status_code == 206:
                total = int(resp.headers.get("Content-Range", "/0").rsplit("/", 1)[-1] or 0)
                data = resp.content
            else:  # Range ignored: stream and keep only the requested bytes
                total = int(resp.headers.get("Content-Length", 0))
                data, got = bytearray(), 0
                for chunk in resp.iter_content(1 << 16):
                    data += chunk
                    got += len(chunk)
                    if got > end:
                        break
                data = bytes(data[start:end + 1])
            return data, name, total
        finally:
            resp.close()

    def file_info(self, file_id):
        if file_id not in self.info:
            _, name, total = self._read(file_id, 0, 0)
            self.info[file_id] = (name, total)
        return self.info[file_id]

    def read_range(self, file_id, start, end):
        data, name, total = self._read(file_id, start, end)
        self.info.setdefault(file_id, (name, total))
        return data


def _filename(disposition):
    m = re.search(r"filename\*=UTF-8''([^;]+)", disposition)
    if m:
        return urllib.parse.unquote(m.group(1))
    m = re.search(r'filename="?([^";]+)"?', disposition)
    return m.group(1) if m else ""


class MockFetcher:
    """Serves Drive ids from a JSON index: {id: {"path": local path} | {"error": kind}}.
    A folder's children get the ids listed in the index under "children"."""

    def __init__(self, index_path):
        with open(index_path) as f:
            self.index = json.load(f)
        self.base = os.path.dirname(os.path.abspath(index_path))

    def _entry(self, fid):
        entry = self.index.get(fid)
        if entry is None:
            raise DriveError("not_found", "Google Drive says this file or folder does not exist")
        if "error" in entry:
            raise DriveError(entry["error"], entry.get("message", f"mock {entry['error']}"))
        return entry

    def _path(self, fid):
        return os.path.join(self.base, self._entry(fid)["path"])

    def list_folder(self, folder_id):
        path = self._path(folder_id)
        if not os.path.isdir(path):
            raise DriveError("unexpected", "Google Drive returned a web page instead of the folder")
        return [{"id": cid, "name": os.path.basename(self.index[cid]["path"]),
                 "is_folder": os.path.isdir(os.path.join(self.base, self.index[cid]["path"]))}
                for cid in self._entry(folder_id).get("children", [])]

    def file_info(self, file_id):
        path = self._path(file_id)
        if os.path.isdir(path):
            raise DriveError("unexpected", "Google Drive returned a web page instead of the file")
        return os.path.basename(path), os.path.getsize(path)

    def read_range(self, file_id, start, end):
        with open(self._path(file_id), "rb") as f:
            f.seek(start)
            return f.read(end - start + 1)


def make_fetcher():
    mock = os.environ.get("AUTOGRADER_DRIVE_MOCK")
    return MockFetcher(mock) if mock else HttpFetcher()


# Remote files: a uniform view of a Drive file or a member of a zip on Drive
class RemoteFile:
    def __init__(self, name, size, read_prefix):
        self.name, self.size, self._read_prefix = name, size, read_prefix

    def read(self, n=None):
        """Return the first n bytes (all bytes if n is None)."""
        return self._read_prefix(self.size if n is None else min(n, self.size))


class RangeReader(io.RawIOBase):
    """Seekable read-only file over a Drive file, fetched in cached blocks (for zipfile)."""
    BLOCK = 1 << 16

    def __init__(self, fetcher, file_id, size):
        self.fetcher, self.file_id, self.size, self.pos, self.cache = fetcher, file_id, size, 0, {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def _block(self, i):
        if i not in self.cache:
            start = i * self.BLOCK
            self.cache[i] = self.fetcher.read_range(self.file_id, start, min(start + self.BLOCK, self.size) - 1)
        return self.cache[i]

    def readinto(self, buf):
        if self.pos >= self.size:
            return 0
        n = min(len(buf), self.size - self.pos)
        out = bytearray()
        while len(out) < n:
            block, off = divmod(self.pos + len(out), self.BLOCK)
            out += self._block(block)[off:off + n - len(out)]
        buf[:n] = out
        self.pos += n
        return n


def _zip_member_reader(zf, info):
    def read_prefix(n):
        with zf.open(info) as f:
            return f.read(n)
    return read_prefix


# Finding the model
MODEL_FILES = ["config.json", "model.safetensors", "tokenizer_config.json", "tokenizer.json", "vocab.json",
               "merges.txt", "generation_config.json", "special_tokens_map.json", "adapter_config.json",
               "adapter_model.safetensors", "pytorch_model.bin"]


def find_model(fetcher, kind, link_id):
    """Locate the saved model behind a Drive link.

    Returns (files, where, log): files maps file name -> RemoteFile for the folder that holds
    config.json (empty if no model folder was found), `where` describes where it was found,
    and `log` lists what was looked at. Raises DriveError if the link itself is not readable.
    """
    log = []
    if kind == "unknown":  # ?id= links: try as a folder, then as a file
        try:
            fetcher.list_folder(link_id)
            kind = "folder"
        except DriveError:
            kind = "file"
    if kind == "file":
        name, size = fetcher.file_info(link_id)
        log.append(f"Link is a file: {name or '(unnamed)'} ({_mb(size)}).")
        if name.lower().endswith(".zip"):
            return _find_in_zip(fetcher, link_id, name, size, log)
        return {}, None, log + ["Share the model folder (or the zip from common.package_submission), "
                                "not a single file."]

    listings, queue, zips = 0, [(link_id, "the shared folder", 0)], []
    while queue and listings < MAX_FOLDER_LISTINGS:
        fid, label, depth = queue.pop(0)
        entries = fetcher.list_folder(fid)
        listings += 1
        names = sorted(e["name"] for e in entries)
        log.append(f"{label}: " + (", ".join(names) if names else "(empty)"))
        files = {e["name"]: e for e in entries if not e["is_folder"]}
        if "config.json" in files or "adapter_config.json" in files:
            out = {}
            for name in MODEL_FILES:
                if name in files:
                    fid_ = files[name]["id"]
                    _, size = fetcher.file_info(fid_)
                    out[name] = RemoteFile(name, size, lambda n, f=fid_: fetcher.read_range(f, 0, n - 1) if n else b"")
            return out, label, log
        zips += [(e, label) for e in files.values() if e["name"].lower().endswith(".zip")]
        if depth + 1 < SEARCH_DEPTH:
            queue += [(e["id"], f"{label}/{e['name']}", depth + 1) for e in entries if e["is_folder"]
                      and not e["name"].startswith(("__MACOSX", "."))]
    if zips:
        zips.sort(key=lambda z: ("submission" not in z[0]["name"].lower(), z[0]["name"]))
        entry, label = zips[0]
        name, size = fetcher.file_info(entry["id"])
        log.append(f"Opening {label}/{name} ({_mb(size)}).")
        return _find_in_zip(fetcher, entry["id"], name, size, log)
    return {}, None, log


def _find_in_zip(fetcher, file_id, name, size, log):
    try:
        zf = zipfile.ZipFile(io.BufferedReader(RangeReader(fetcher, file_id, size), 1 << 16))
    except zipfile.BadZipFile:
        return {}, None, log + [f"{name} is not a valid zip file (it may be incomplete)."]
    infos = [i for i in zf.infolist() if not i.is_dir() and "__MACOSX" not in i.filename]
    dirs = {}
    for i in infos:
        d, base = os.path.split(i.filename)
        dirs.setdefault(d, {})[base] = i
    candidates = [d for d, f in dirs.items() if "config.json" in f or "adapter_config.json" in f]
    if not candidates:
        return {}, None, log + [f"{name} has no config.json. Files: " + ", ".join(sorted(i.filename for i in infos)[:30])]
    d = sorted(candidates, key=lambda d: ("model.safetensors" not in dirs[d], len(d)))[0]
    log.append(f"Found the model in {name} at {d or '(top level)'}/.")
    files = {n: RemoteFile(n, i.file_size, _zip_member_reader(zf, i)) for n, i in dirs[d].items() if n in MODEL_FILES}
    return files, f"{name}:{d or '/'}", log


def _mb(n):
    return f"{n / 1e6:.1f} MB" if n >= 1e5 else f"{n:,} bytes"


if __name__ == "__main__":  # staff debugging: python drive.py "<drive link>"
    import sys
    parsed = parse_link(" ".join(sys.argv[1:]))
    if not parsed:
        sys.exit("No Drive link found.")
    found, where, lines = find_model(make_fetcher(), parsed[0], parsed[1])
    print("\n".join(lines))
    for n, f in found.items():
        print(f"  {n}: {_mb(f.size)}")
