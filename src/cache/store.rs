//! CacheStore — pluggable persistence backends for per-user KV caches.
//!
//! The hippocampus serves as personal memory for each user. Caches are
//! serialized to storage between sessions and loaded on demand.

use std::io;
use std::path::{Path, PathBuf};

/// Pluggable storage backend for user KV caches.
pub trait CacheStore {
    /// Load a user's serialized KV cache. Returns None if no history exists.
    fn load(&self, user_id: &str) -> io::Result<Option<Vec<u8>>>;

    /// Persist a user's serialized KV cache.
    fn save(&self, user_id: &str, data: &[u8]) -> io::Result<()>;

    /// Delete a user's cache (account deletion / GDPR).
    fn delete(&self, user_id: &str) -> io::Result<()>;

    /// Check if a user has a stored cache.
    fn exists(&self, user_id: &str) -> io::Result<bool>;
}

/// File-based cache store: one file per user in a directory.
///
/// Layout: `{base_dir}/{user_id}.engram`
pub struct FileCacheStore {
    base_dir: PathBuf,
}

impl FileCacheStore {
    /// Create a new file-based store. Creates the directory if it doesn't exist.
    pub fn new(base_dir: impl AsRef<Path>) -> io::Result<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&base_dir)?;
        Ok(Self { base_dir })
    }

    fn user_path(&self, user_id: &str) -> PathBuf {
        self.base_dir.join(format!("{user_id}.engram"))
    }
}

impl CacheStore for FileCacheStore {
    fn load(&self, user_id: &str) -> io::Result<Option<Vec<u8>>> {
        let path = self.user_path(user_id);
        match std::fs::read(&path) {
            Ok(data) => Ok(Some(data)),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }

    fn save(&self, user_id: &str, data: &[u8]) -> io::Result<()> {
        let path = self.user_path(user_id);
        // Atomic write: temp file + rename.
        let tmp = path.with_extension("engram.tmp");
        std::fs::write(&tmp, data)?;
        std::fs::rename(&tmp, &path)?;
        Ok(())
    }

    fn delete(&self, user_id: &str) -> io::Result<()> {
        let path = self.user_path(user_id);
        match std::fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e),
        }
    }

    fn exists(&self, user_id: &str) -> io::Result<bool> {
        Ok(self.user_path(user_id).exists())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_store_roundtrip() {
        let dir = std::env::temp_dir().join("engram_test_store");
        let _ = std::fs::remove_dir_all(&dir);
        let store = FileCacheStore::new(&dir).unwrap();

        assert!(!store.exists("alice").unwrap());
        assert!(store.load("alice").unwrap().is_none());

        store.save("alice", b"test data").unwrap();
        assert!(store.exists("alice").unwrap());

        let loaded = store.load("alice").unwrap().unwrap();
        assert_eq!(loaded, b"test data");

        store.delete("alice").unwrap();
        assert!(!store.exists("alice").unwrap());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn delete_nonexistent_is_ok() {
        let dir = std::env::temp_dir().join("engram_test_store_del");
        let _ = std::fs::remove_dir_all(&dir);
        let store = FileCacheStore::new(&dir).unwrap();

        // Should not error.
        store.delete("nobody").unwrap();

        let _ = std::fs::remove_dir_all(&dir);
    }
}
