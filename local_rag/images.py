"""Explicitly curated image memory. Derivatives share the image row and scope.

No attachment listener, temporary attachment store, or conversation ingestion.
OCR is local Tesseract when installed; absence is reported, never invented.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import threading
import time
import tempfile
from array import array

from PIL import Image
from .store import MemoryStore
from .policy import classify_text, IngestDecision


# Metadata/derivatives only: originals are read explicitly for recovery/integrity.
_IMAGE_COLUMNS = ('id,namespace,hash,filename,scope,scope_key,version_group,active,'
                  'description,ocr,ocr_status,reason,clip,text_vector,text_dimensions,created')


class ImagePolicyError(ValueError):
    pass


def screen_image_text(*values):
    # Short labels are valid image metadata; BLOCK matches text-memory secrets
    # and injection screening without imposing its prose-length heuristic.
    if any(classify_text(value) is IngestDecision.BLOCK for value in values):
        raise ImagePolicyError('Image rejected by ingestion policy')


def local_ocr(path):
    executable = shutil.which('tesseract')
    import sys
    if executable:
        command, status = [executable, str(path), 'stdout'], 'tesseract'
    elif sys.platform == 'darwin' and shutil.which('swift'):
        cache = Path(path).parent.parent / '.cache' / 'swift-ocr'
        cache.mkdir(parents=True, exist_ok=True)
        command = [shutil.which('swift'), '-module-cache-path', str(cache), str(Path(__file__).with_name('ocr.swift')), str(path)]
        status = 'apple-vision'
    else:
        return '', 'unavailable: install tesseract and required language packs (or macOS Swift tools)'
    try:
        # Drain at most limit+1 bytes, not an unbounded capture or temporary file.
        # Reject overflow BEFORE stripping/decoding: unseen text may contain secrets.
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        stdout = process.stdout
        assert stdout is not None
        output = []
        def read_output():
            try:
                output.append(stdout.read(32001))
            except OSError:
                pass  # Empty output list fails closed below.
        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        try:
            reader.join(timeout=30)
            if reader.is_alive():
                raise ImagePolicyError('Image rejected: OCR output timed out')
            if not output or len(output[0]) > 32000:
                raise ImagePolicyError('Image rejected: OCR output exceeds 32000 byte limit')
            if process.wait(timeout=30):
                raise subprocess.CalledProcessError(process.returncode, command)
            return output[0].decode('utf-8').strip(), status
        finally:
            if reader.is_alive() or process.poll() is None:
                process.kill()
            process.wait()
            reader.join()
            stdout.close()
    except (OSError, subprocess.SubprocessError) as exc:
        return '', f'failed: {type(exc).__name__}'


class CuratedImages:
    def __init__(self, home, *, text, visual, ocr=local_ocr):
        self.directory = Path(home) / 'local-rag' / 'images'
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.directory.is_symlink():
            raise ValueError('Managed images directory must not be a symlink')
        self.directory.chmod(0o700)
        self.text, self.visual, self.ocr = text, visual, ocr
        self.lock = threading.RLock()
        database = self.directory.parent / 'curated-images.db'
        fd = os.open(database, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            os.fchmod(fd, 0o600)
        finally:
            os.close(fd)
        self.db = sqlite3.connect(database, check_same_thread=False)
        self.db.execute('PRAGMA secure_delete=ON')
        self.db.row_factory = sqlite3.Row
        with self.db:
            self.db.execute('CREATE TABLE IF NOT EXISTS image_gc (filename TEXT PRIMARY KEY)')
            self.db.execute('CREATE TABLE IF NOT EXISTS image_checks (id INTEGER PRIMARY KEY, checked REAL NOT NULL)')
            self.db.execute('CREATE TABLE IF NOT EXISTS image_maintenance (slot INTEGER PRIMARY KEY, checked REAL NOT NULL)')
            self.db.execute('''CREATE TABLE IF NOT EXISTS curated_images (
                id INTEGER PRIMARY KEY, namespace TEXT NOT NULL, hash TEXT NOT NULL,
                filename TEXT NOT NULL, scope TEXT NOT NULL, scope_key TEXT NOT NULL,
                version_group TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1,
                description TEXT NOT NULL, ocr TEXT NOT NULL, ocr_status TEXT NOT NULL,
                reason TEXT NOT NULL, clip BLOB NOT NULL, text_vector BLOB NOT NULL,
                text_dimensions INTEGER NOT NULL, created REAL NOT NULL,
                UNIQUE(namespace,scope,scope_key,version_group,hash))''')
        self._migrate_originals()

    def _migrate_originals(self):
        # Schema and backfill are one writer transaction, including legacy failure.
        self.db.execute('BEGIN IMMEDIATE')
        try:
            columns = {r[1] for r in self.db.execute('PRAGMA table_info(curated_images)')}
            if 'original' not in columns:
                self.db.execute('ALTER TABLE curated_images ADD COLUMN original BLOB')
            for row in self.db.execute('SELECT * FROM curated_images WHERE original IS NULL').fetchall():
                self._validate_filename(row)
                try:
                    data = self._read_managed(self.directory / row['filename'])
                    self._validate_original(row, data)
                except (OSError, ValueError) as exc:
                    raise ValueError(f"Legacy image {row['id']} has no verified original; restore its intact managed file before retrying migration") from exc
                self.db.execute('UPDATE curated_images SET original=? WHERE id=?', (data, row['id']))
            self.db.commit()
        except BaseException:
            self.db.rollback()
            self.db.close()
            raise

    @staticmethod
    def _validate_filename(row):
        if not re.fullmatch(r'[0-9a-f]{64}\.(png|jpg|webp|bmp)', row['filename']) or not row['filename'].startswith(row['hash'] + '.'):
            raise ValueError('Managed image filename integrity failure')

    def _validate_original(self, row, data):
        self._validate_filename(row)
        if not isinstance(data, bytes) or hashlib.sha256(data).hexdigest() != row['hash'] or row['hash'] + self._suffix(data) != row['filename']:
            raise ValueError('Stored original integrity failure')

    def _private_cache(self):
        cache = self.directory.parent / '.cache'
        cache.mkdir(mode=0o700, exist_ok=True)
        if cache.is_symlink():
            raise ValueError('Image staging directory must not be a symlink')
        cache.chmod(0o700)
        return cache

    def _atomic_file(self, target, data):
        # Private staging is excluded from native backup; final name is never partial.
        cache = self._private_cache()
        fd, name = tempfile.mkstemp(dir=cache, prefix='image-')
        try:
            with os.fdopen(fd, 'wb') as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, target)
        finally:
            Path(name).unlink(missing_ok=True)

    def materialize(self):
        """Explicit recovery only: DB wins over missing/edited filesystem copies.

        Run before opening a restored home in a provider (normal reconciliation
        treats missing managed files as user deletions). Never runs inference.
        """
        with self.lock:
            self.db.execute('BEGIN IMMEDIATE')
            try:
                rows = self.db.execute('SELECT * FROM curated_images').fetchall()
                # Validate the entire snapshot before writing any recovered file.
                for row in rows:
                    self._validate_original(row, row['original'])
                for row in rows:
                    self._atomic_file(self.directory / row['filename'], row['original'])
                self.db.commit()
                return len(rows)
            except BaseException:
                self.db.rollback()
                raise

    @staticmethod
    def _suffix(data):
        if len(data) > 25_000_000:
            raise ValueError('Image exceeds 25 MB limit')
        import io
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height > 40_000_000:
                raise ValueError('Image exceeds 40 megapixel limit')
            suffix = {'PNG': '.png', 'JPEG': '.jpg', 'WEBP': '.webp', 'BMP': '.bmp'}.get(image.format)
            if not suffix:
                raise ValueError('Unsupported image format')
            image.verify()
        return suffix

    @staticmethod
    def scope_key(scope, project, session):
        if scope not in {'global', 'project', 'session'}:
            raise ValueError('Invalid image scope')
        key = project if scope == 'project' else session if scope == 'session' else ''
        if scope != 'global' and not key:
            raise ValueError(f'{scope} image requires active scope')
        return key

    def save(self, namespace, path, *, decision, reason, scope, project='', session='', group='', description=''):
        if decision != 'save' or not isinstance(reason, str) or not reason.strip():
            raise ValueError('Image requires explicit curated save decision and reason')
        if not isinstance(group, str) or len(group) > 200 or not isinstance(description, str) or len(description) > 4000 or len(reason) > 1000:
            raise ValueError('Invalid image metadata')
        screen_image_text(description, reason, group)
        key = self.scope_key(scope, project, session)
        path = Path(path)
        data = path.read_bytes()
        if len(data) > 25_000_000:
            raise ValueError('Image exceeds 25 MB limit')
        digest = hashlib.sha256(data).hexdigest()
        group = group.strip() or digest
        suffix = self._suffix(data)
        filename = digest + suffix
        managed = self.directory / filename
        # BEGIN IMMEDIATE serializes saves/deletes across processes, including GC.
        with self.lock:
            self.db.execute('BEGIN IMMEDIATE')
            created_file = False
            try:
                old = self.db.execute(f'SELECT {_IMAGE_COLUMNS} FROM curated_images WHERE namespace=? AND scope=? AND scope_key=? AND version_group=? AND hash=?', (namespace, scope, key, group, digest)).fetchone()
                if old:
                    self.db.commit()
                    return self._result(old)
                with tempfile.NamedTemporaryFile(dir=self._private_cache(), suffix=suffix) as snapshot:
                    snapshot.write(data)
                    snapshot.flush()
                    ocr, status, clip, vector = self._derivatives(Path(snapshot.name), description)
                if not managed.exists():
                    self._atomic_file(managed, data)
                    created_file = True
                elif managed.is_symlink() or self._read_managed(managed) != data:
                    raise ValueError('Managed image integrity failure')
                self.db.execute('UPDATE curated_images SET active=0 WHERE namespace=? AND scope=? AND scope_key=? AND version_group=?', (namespace, scope, key, group))
                cur = self.db.execute('''INSERT INTO curated_images(namespace,hash,filename,scope,scope_key,version_group,description,ocr,ocr_status,reason,clip,text_vector,text_dimensions,created,original)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', (namespace,digest,filename,scope,key,group,description,ocr,status,reason,array('f',clip).tobytes(),array('f',vector).tobytes(),self.text.dimensions,time.time(),data))
                row = self.db.execute(f'SELECT {_IMAGE_COLUMNS} FROM curated_images WHERE id=?', (cur.lastrowid,)).fetchone()
                self.db.commit()
                return self._result(row)
            except BaseException:
                try:
                    if created_file:
                        managed.unlink(missing_ok=True)
                finally:
                    self.db.rollback()
                raise

    def reindex(self, namespace, image_id, *, project='', session='', description=None):
        """Explicit scoped derivative refresh; does not revive missing/edited files."""
        if description is not None and (not isinstance(description, str) or len(description) > 4000):
            raise ValueError('Invalid image description')
        with self.lock:
            self.db.execute('BEGIN IMMEDIATE')
            try:
                row = self.db.execute(f"""SELECT {_IMAGE_COLUMNS} FROM curated_images WHERE namespace=? AND id=?
                    AND (scope='global' OR (scope='project' AND scope_key=?) OR (scope='session' AND scope_key=?))""",
                    (namespace,image_id,project,session)).fetchone()
                if row is None:
                    self.db.rollback()
                    return None
                data = self._read_managed(self.directory / row['filename'])
                self._validate_original(row, data)
                description = row['description'] if description is None else description
                screen_image_text(description, row['reason'], row['version_group'])
                cache = self._private_cache()
                with tempfile.NamedTemporaryFile(dir=cache, suffix=Path(row['filename']).suffix) as snapshot:
                    snapshot.write(data)
                    snapshot.flush()
                    ocr, status, clip, vector = self._derivatives(Path(snapshot.name), description)
                self.db.execute("""UPDATE curated_images SET description=?,ocr=?,ocr_status=?,
                    clip=?,text_vector=?,text_dimensions=?,original=? WHERE id=?""",
                    (description,ocr,status,array('f',clip).tobytes(),array('f',vector).tobytes(),self.text.dimensions,data,image_id))
                result = self._result(self.db.execute(f'SELECT {_IMAGE_COLUMNS} FROM curated_images WHERE id=?', (image_id,)).fetchone())
                self.db.commit()
                return result
            except BaseException:
                self.db.rollback()
                raise

    def _derivatives(self, path, description):
        ocr, status = self.ocr(path)
        if status.startswith(('failed', 'unavailable')):
            # Unchecked pixels must never reach embeddings or durable storage.
            # RuntimeError is retryable in reconcile, unlike a policy rejection:
            # preserve the last verified original/index when OCR is down.
            raise RuntimeError('Image OCR failed or unavailable; retry when OCR is working')
        screen_image_text(description, ocr)
        derivative = '\n'.join(s for s in (description, ocr) if s)
        clip = self.visual().embed_image(path)
        vector = self.text.embed_document(derivative) if derivative else [0.] * self.text.dimensions
        if len(clip) != 512 or len(vector) != self.text.dimensions:
            raise ValueError('Image embedding dimension mismatch')
        return ocr, status, clip, vector

    @staticmethod
    def _read_managed(path):
        # Do not follow replacement symlinks or read unbounded replacement files.
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as handle:
            import stat
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise ValueError('Managed image must be a regular file')
            data = handle.read(25_000_001)
        if len(data) > 25_000_000:
            raise ValueError('Image exceeds 25 MB limit')
        return data

    def _drop_row(self, row):
        self.db.execute('DELETE FROM curated_images WHERE id=?', (row['id'],))
        self.db.execute('DELETE FROM image_checks WHERE id=?', (row['id'],))
        self.db.execute('INSERT OR IGNORE INTO image_gc VALUES(?)', (row['filename'],))

    def reconcile(self, namespace, *, project='', session='', force=False):
        """Check one already-curated row per minute, shared across processes.

        Never scans/adopts files. Managed originals are durable snapshots: external
        source disappearance is irrelevant. In-place managed edits replace their
        invalidated derivatives, preserving the row's scope, group and active bit.
        Explicit saves remain the way to retain an intact previous version.
        ``force`` is an internal test/maintenance override, not a model tool arg.
        """
        with self.lock:
            self.db.execute('BEGIN IMMEDIATE')
            created = []
            try:
                now = time.time()
                last = self.db.execute('SELECT checked FROM image_maintenance WHERE slot=1').fetchone()
                if not force and last and 0 <= now - last['checked'] < 60:
                    self.db.rollback()
                    return
                row = self.db.execute('''SELECT i.id,i.namespace,i.hash,i.filename,i.scope,i.scope_key,
                    i.version_group,i.description,i.reason FROM curated_images i
                    LEFT JOIN image_checks c ON c.id=i.id
                    WHERE namespace=? AND (scope='global' OR (scope='project' AND scope_key=?)
                    OR (scope='session' AND scope_key=?))
                    ORDER BY coalesce(c.checked,0), i.id LIMIT 1''', (namespace,project,session)).fetchone()
                self.db.execute('INSERT OR REPLACE INTO image_maintenance VALUES(1,?)', (now,))
                if row:
                    self.db.execute('INSERT OR REPLACE INTO image_checks VALUES(?,?)', (row['id'],now))
                    try:
                        self._reconcile_row(row, created)
                    except (OSError, RuntimeError):
                        # Temporary OCR/inference/IO failures retry on a later pass.
                        # Clean partial writes before releasing SQLite's writer lock.
                        for path in created:
                            path.unlink(missing_ok=True)
                        created.clear()
                        # Retrieval independently rejects mismatched bytes meanwhile.
                self.db.commit()
            except BaseException:
                # Keep the writer lock until new files are removed: another process
                # must not acquire a reference between rollback and cleanup.
                try:
                    for path in created:
                        path.unlink(missing_ok=True)
                finally:
                    self.db.rollback()
                raise
            self._collect_garbage()

    def _reconcile_row(self, row, created):
        path = self.directory / row['filename']
        try:
            if path.is_symlink():
                self._drop_row(row)
                return
            data = self._read_managed(path)
            suffix = self._suffix(data)
        except (FileNotFoundError, ValueError, Image.UnidentifiedImageError):
            self._drop_row(row)
            return
        digest = hashlib.sha256(data).hexdigest()
        if digest == row['hash']:
            return
        # OCR reads a private stable snapshot, never an arbitrary new directory file.
        cache = self._private_cache()
        with tempfile.NamedTemporaryFile(dir=cache, suffix=suffix) as snapshot:
            snapshot.write(data)
            snapshot.flush()
            try:
                screen_image_text(row['description'], row['reason'], row['version_group'])
                ocr, status, clip, vector = self._derivatives(Path(snapshot.name), row['description'])
            except ImagePolicyError:
                self._drop_row(row)
                return
        filename = digest + suffix
        target = self.directory / filename
        if not target.exists():
            self._atomic_file(target, data)
            created.append(target)
        elif self._read_managed(target) != data:
            raise RuntimeError('Managed image integrity failure')
        duplicate = self.db.execute('''SELECT id FROM curated_images WHERE namespace=?
            AND scope=? AND scope_key=? AND version_group=? AND hash=? AND id!=?''',
            (row['namespace'],row['scope'],row['scope_key'],row['version_group'],digest,row['id'])).fetchone()
        if duplicate:
            # Never reactivate a historical version through deduplication.
            self._drop_row(row)
        else:
            self.db.execute('''UPDATE curated_images SET hash=?,filename=?,ocr=?,ocr_status=?,
                clip=?,text_vector=?,text_dimensions=?,original=? WHERE id=?''',
                (digest,filename,ocr,status,array('f',clip).tobytes(),array('f',vector).tobytes(),self.text.dimensions,data,row['id']))
            self.db.execute('INSERT OR IGNORE INTO image_gc VALUES(?)', (row['filename'],))

    def _result(self, row, score=None):
        result = {key: row[key] for key in ('id', 'scope', 'version_group', 'active', 'description', 'ocr', 'ocr_status')}
        result.update(path=str(self.directory / row['filename']), sha256=row['hash'], kind='image', source=f"image:{row['id']}", text='\n'.join(x for x in (row['description'], row['ocr']) if x))
        if score is not None:
            result['score'] = score
        return result

    def search(self, namespace, query, *, project='', session='', limit=5, include_history=False, reconcile=True):
        if reconcile:
            self.reconcile(namespace, project=project, session=session)
        with self.lock:
            rows = self.db.execute(f'''SELECT {_IMAGE_COLUMNS} FROM curated_images WHERE namespace=?
                AND (scope='global' OR (scope='project' AND scope_key=?) OR (scope='session' AND scope_key=?))'''
                + ('' if include_history else ' AND active=1'), (namespace,project,session)).fetchall()
        if not rows:
            return []
        qtext = self.text.embed_query(query)
        qclip = self.visual().embed_text(query)
        terms = set(re.findall(r'\w+', query.casefold()))
        results = []
        for row in rows:
            if not (self.directory / row['filename']).is_file():
                continue
            if row['text_dimensions'] != len(qtext):
                raise ValueError('Image text dimensions changed; explicitly reindex images')
            derivative = row['description'] + '\n' + row['ocr']
            lexical = len(terms & set(re.findall(r'\w+', derivative.casefold()))) / max(1, len(terms))
            semantic = MemoryStore._cosine(qtext, array('f', row['text_vector']))
            visual = MemoryStore._cosine(qclip, array('f', row['clip']))
            score = max(.75 * semantic + .25 * lexical, visual, .8 * lexical)
            if score >= .25:
                results.append(self._result(row, score))
        # Verify only the bounded top candidates. Pending maintenance or failed
        # reindex must never expose changed/unchecked pixels via stale derivatives.
        verified = []
        for result in sorted(results, key=lambda r: r['score'], reverse=True)[:max(0, min(20, limit))]:
            try:
                if hashlib.sha256(self._read_managed(result['path'])).hexdigest() == result['sha256']:
                    verified.append(result)
            except (OSError, ValueError):
                pass
        return verified

    def delete(self, namespace, image_id, *, project='', session=''):
        with self.lock:
            self.db.execute('BEGIN IMMEDIATE')
            try:
                row = self.db.execute(f'''SELECT {_IMAGE_COLUMNS} FROM curated_images WHERE namespace=? AND id=?
                    AND (scope='global' OR (scope='project' AND scope_key=?) OR (scope='session' AND scope_key=?))''',
                    (namespace,image_id,project,session)).fetchone()
                if row is None:
                    self.db.rollback()
                    return False
                self._drop_row(row)
                self.db.commit()
                self._collect_garbage()
                return True
            except BaseException:
                self.db.rollback()
                raise

    def _collect_garbage(self, limit=8):
        # A second write transaction prevents a concurrent save between checking
        # references and unlinking. The row deletion has ALREADY committed. A
        # crash/unlink failure leaves a durable retry, never a restored dangling row.
        with self.lock:
            self.db.execute('BEGIN IMMEDIATE')
            try:
                for row in self.db.execute('SELECT filename FROM image_gc LIMIT ?', (limit,)).fetchall():
                    filename = row['filename']
                    if not self.db.execute('SELECT 1 FROM curated_images WHERE filename=?', (filename,)).fetchone():
                        try:
                            (self.directory / filename).unlink(missing_ok=True)
                        except OSError:
                            continue
                    self.db.execute('DELETE FROM image_gc WHERE filename=?', (filename,))
                self.db.commit()
            except BaseException:
                self.db.rollback()
                raise

    def count(self, namespace):
        with self.lock:
            return self.db.execute('SELECT count(*) FROM curated_images WHERE namespace=?', (namespace,)).fetchone()[0]

    def close(self):
        with self.lock:
            self.db.close()
