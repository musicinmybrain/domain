use core::{
    borrow::{Borrow, BorrowMut},
    cmp, fmt,
    hash::{Hash, Hasher},
    iter,
    ops::{Deref, DerefMut},
};

use super::{Label, Labels, Name, RelName};

/// A domain name that is absolute or relative.
#[repr(transparent)]
pub struct UncertainName([u8]);

impl UncertainName {
    /// The maximum size of an uncertain domain name in the wire format.
    pub const MAX_SIZE: usize = 255;

    /// The root name.
    pub const ROOT: &'static Self =
        unsafe { Self::from_bytes_unchecked(&[0u8]) };
}

impl UncertainName {
    /// Assume a byte string is a valid [`UncertainName`].
    ///
    /// # Safety
    ///
    /// The byte string must be correctly encoded in the wire format, and
    /// within the size restriction (255 bytes or fewer).  If it is 255 bytes
    /// long, it must end with a root label.
    pub const unsafe fn from_bytes_unchecked(bytes: &[u8]) -> &Self {
        // SAFETY: 'UncertainName' is a 'repr(transparent)' wrapper around
        // '[u8]', so casting a '[u8]' into an 'UncertainName' is sound.
        core::mem::transmute(bytes)
    }

    /// Assume a mutable byte string is a valid [`UncertainName`].
    ///
    /// # Safety
    ///
    /// The byte string must be correctly encoded in the wire format, and
    /// within the size restriction (255 bytes or fewer).  If it is 255 bytes
    /// long, it must end with a root label.
    pub unsafe fn from_bytes_unchecked_mut(bytes: &mut [u8]) -> &mut Self {
        // SAFETY: 'UncertainName' is a 'repr(transparent)' wrapper around
        // '[u8]', so casting a '[u8]' into an 'UncertainName' is sound.
        core::mem::transmute(bytes)
    }

    /// Try converting a byte string into a [`UncertainName`].
    ///
    /// The byte string is confirmed to be correctly encoded in the wire
    /// format.  If it is not properly encoded, an error is returned.
    ///
    /// Runtime: `O(bytes.len())`.
    pub fn from_bytes(bytes: &[u8]) -> Result<&Self, UncertainNameError> {
        if bytes.len() > Name::MAX_SIZE {
            // Absolute or relative, this domain name is too long.
            return Err(UncertainNameError);
        }

        // Iterate through labels in the name.
        let mut index = 0usize;
        while index < bytes.len() {
            let length = bytes[index];
            if length == 0 {
                // Assume this was the end of the name.
                index += 1;
                break;
            } else if length >= 64 {
                // An invalid label length (or a compression pointer).
                return Err(UncertainNameError);
            } else {
                // This was the length of the label, excluding the length
                // octet.
                index += 1 + length as usize;
            }
        }

        // We must land exactly at the end of the name, otherwise there was an
        // empty label in the middle of the name, or the previous label
        // reported a length that was too long.
        if index != bytes.len() {
            return Err(UncertainNameError);
        }

        // SAFETY: 'bytes' has been confirmed to be correctly encoded.
        Ok(unsafe { Self::from_bytes_unchecked(bytes) })
    }
}

impl UncertainName {
    /// The size of this name in the wire format.
    pub const fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether this name contains no labels at all.
    pub const fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Whether this is the root label.
    pub const fn is_root(&self) -> bool {
        self.0.len() == 1
    }

    /// The wire format representation of the name.
    pub const fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// The parent of this name, if any.
    ///
    /// The name containing all but the first label is returned.  If this is a
    /// root name, [`None`] is returned.
    ///
    /// Runtime: `O(1)`.
    pub fn parent(&self) -> Option<&Self> {
        if self.is_empty() || self.is_root() {
            return None;
        }

        let bytes = self.as_bytes();
        let bytes = &bytes[1 + bytes[0] as usize..];

        // SAFETY: 'bytes' is 253 bytes or smaller and has valid labels.
        Some(unsafe { Self::from_bytes_unchecked(bytes) })
    }

    /// The labels in this name.
    ///
    /// The root label is included in the iterator.
    ///
    /// Runtime: `O(1)`.  Each step of the iterator has runtime `O(1)` too.
    pub const fn labels(&self) -> Labels<'_> {
        // SAFETY: This is a valid absolute or relative name.
        unsafe { Labels::from_bytes_unchecked(self.as_bytes()) }
    }

    /// Whether this name starts with a particular relative name.
    ///
    /// Runtime: `O(prefix.len())`, which is less than `O(self.len())`.
    pub fn starts_with(&self, prefix: &RelName) -> bool {
        if self.len() < prefix.len() {
            return false;
        }

        // Label lengths are never ASCII characters, because they start from
        // byte value 65.  So we can treat the byte strings as ASCII.
        self.as_bytes()[..prefix.len()]
            .eq_ignore_ascii_case(prefix.as_bytes())
    }

    /// Whether this name ends with a particular absolute name.
    ///
    /// Runtime: `O(self.len())`, which is more than `O(suffix.len())`.
    pub fn ends_with(&self, suffix: &Self) -> bool {
        if self.len() < suffix.len() {
            return false;
        }

        // We want to compare the last bytes of the current name to the given
        // candidate.  To do so, we need to ensure that those last bytes start
        // at a valid label boundary.

        let mut index = 0usize;
        let offset = self.len() - suffix.len();
        while index < offset {
            index += 1 + self.0[index] as usize;
        }

        if index != offset {
            return false;
        }

        // Label lengths are never ASCII characters, because they start from
        // byte value 65.  So we can treat the byte strings as ASCII.
        self.as_bytes()[offset..].eq_ignore_ascii_case(suffix.as_bytes())
    }
}

impl UncertainName {
    /// Split this name into a label and the rest.
    ///
    /// If this is the root name, [`None`] is returned.  The returned label
    /// will always be non-empty.
    ///
    /// Runtime: `O(1)`.
    pub fn split_first(&self) -> Option<(&Label, &Self)> {
        if self.is_empty() || self.is_root() {
            return None;
        }

        let bytes = self.as_bytes();
        let (label, rest) = bytes[1..].split_at(bytes[0] as usize);

        // SAFETY: 'self' only contains valid labels.
        let label = unsafe { Label::from_bytes_unchecked(label) };
        // SAFETY: 'rest' is 253 bytes or smaller and has valid labels.
        let rest = unsafe { Self::from_bytes_unchecked(rest) };

        Some((label, rest))
    }

    /// Strip a prefix from this name.
    ///
    /// If this name has the given prefix (see [`Self::starts_with()`]), the
    /// rest of the name without the prefix is returned.  Otherwise, [`None`]
    /// is returned.
    ///
    /// Runtime: `O(prefix.len())`, which is less than `O(self.len())`.
    pub fn strip_prefix<'a>(&'a self, prefix: &RelName) -> Option<&'a Self> {
        if self.starts_with(prefix) {
            let bytes = &self.as_bytes()[prefix.len()..];

            // SAFETY: 'self' and 'prefix' consist of whole labels, and 'self'
            // start with the same labels as 'prefix'; removing those labels
            // still leaves 'self' with whole labels.
            Some(unsafe { Self::from_bytes_unchecked(bytes) })
        } else {
            None
        }
    }

    /// Strip a suffix from this name.
    ///
    /// If this name has the given suffix (see [`Self::ends_with()`]), the
    /// rest of the name without the suffix is returned.  Otherwise, [`None`]
    /// is returned.
    ///
    /// Runtime: `O(self.len())`, which is more than `O(suffix.len())`.
    pub fn strip_suffix<'a>(&'a self, suffix: &Self) -> Option<&'a Self> {
        if self.ends_with(suffix) {
            let bytes = &self.as_bytes()[..self.len() - suffix.len()];

            // SAFETY: 'self' and 'suffix' consist of whole labels, and 'self'
            // ended with the same labels as 'suffix'; removing those labels
            // still leaves 'self' with whole labels.
            Some(unsafe { Self::from_bytes_unchecked(bytes) })
        } else {
            None
        }
    }

    /// Canonicalize this domain name.
    ///
    /// All uppercase ASCII characters in the name will be lowercased.
    ///
    /// Runtime: `O(self.len())`.
    pub fn canonicalize(&mut self) {
        // Label lengths are never ASCII characters, because they start from
        // byte value 65.  So we can treat the entire byte string as ASCII.
        self.0.make_ascii_lowercase()
    }
}

impl PartialEq for UncertainName {
    /// Compare labels by their canonical value.
    ///
    /// Canonicalized labels have uppercase ASCII characters lowercased, so
    /// this function compares the two names case-insensitively.
    ///
    // Runtime: `O(self.len())`, which is equal to `O(that.len())`.
    fn eq(&self, that: &Self) -> bool {
        // Label lengths are never ASCII characters, because they start from
        // byte value 65.  So we can treat the entire byte string as ASCII.
        self.0.eq_ignore_ascii_case(&that.0)
    }
}

impl Eq for UncertainName {}

impl PartialOrd for UncertainName {
    /// Compare names according to the canonical ordering.
    ///
    /// The 'canonical DNS name order' is defined in RFC 4034, section 6.1.
    /// Essentially, any shared suffix of labels is stripped away, and the
    /// remaining unequal label at the end is compared case-insensitively.
    /// Absolute domain names come before relative ones.
    ///
    /// Runtime: `O(self.len() + that.len())`.
    fn partial_cmp(&self, that: &Self) -> Option<cmp::Ordering> {
        Some(Ord::cmp(self, that))
    }
}

impl Ord for UncertainName {
    /// Compare names according to the canonical ordering.
    ///
    /// The 'canonical DNS name order' is defined in RFC 4034, section 6.1.
    /// Essentially, any shared suffix of labels is stripped away, and the
    /// remaining unequal label at the end is compared case-insensitively.
    /// Absolute domain names come before relative ones.
    ///
    /// Runtime: `O(self.len() + that.len())`.
    fn cmp(&self, that: &Self) -> cmp::Ordering {
        // We want to find a shared suffix between the two names, and the
        // labels immediately before that shared suffix.  However, we can't
        // determine label boundaries when working backward.  So, we find a
        // shared suffix (even if it crosses partially between labels), then
        // iterate through both names until we find their label boundaries up
        // to the suffix.

        let this_iter = self.as_bytes().iter().rev();
        let that_iter = that.as_bytes().iter().rev();
        let suffix = iter::zip(this_iter, that_iter)
            .position(|(l, r)| !l.eq_ignore_ascii_case(r));

        if let Some(suffix) = suffix {
            // Iterate through the labels in both names until both have a tail
            // of equal size within the shared suffix we found.

            // SAFETY: At least one unequal byte exists in both names, and it
            // cannot be the root label, so there must be at least one
            // non-root label in both names.
            let (mut this_head, mut this_tail) =
                unsafe { self.split_first().unwrap_unchecked() };
            let (mut that_head, mut that_tail) =
                unsafe { self.split_first().unwrap_unchecked() };

            loop {
                let (this_len, that_len) = (this_tail.len(), that_tail.len());

                if this_len == that_len && this_len < suffix {
                    // We have found the shared suffix of labels.  Now, we
                    // must have two unequal head labels; we compare them
                    // (ASCII case insensitively).
                    break Ord::cmp(this_head, that_head);
                }

                // If one tail is longer than the other, it will be shortened.
                // Any tail longer than the suffix will also be shortened.

                if this_len > that_len || this_len > suffix {
                    // SAFETY: 'this_tail' has strictly more than one byte.
                    (this_head, this_tail) =
                        unsafe { this_tail.split_first().unwrap_unchecked() };
                }

                if that_len > this_len || that_len > suffix {
                    // SAFETY: 'that_tail' has strictly more than one byte.
                    (that_head, that_tail) =
                        unsafe { that_tail.split_first().unwrap_unchecked() };
                }
            }
        } else {
            // The shorter name is a suffix of the longer one.  If the names
            // are of equal length, they are equal; otherwise, the longer one
            // has more labels, and is greater than the shorter one.
            Ord::cmp(&self.len(), &that.len())
        }
    }
}

impl Hash for UncertainName {
    /// Hash this label by its canonical value.
    ///
    /// The hasher is provided with the labels in this name with ASCII
    /// characters lowercased.  Each label is preceded by its length as `u8`.
    ///
    /// The same scheme is used by [`RelName`] and [`Label`], so a tuple of
    /// any of these types will have the same hash as the concatenation of the
    /// labels.
    ///
    /// Runtime: `O(self.len())`.
    fn hash<H: Hasher>(&self, state: &mut H) {
        // NOTE: Label lengths are not affected by 'to_ascii_lowercase()'
        // since they are always less than 64.  As such, we don't need to
        // iterate over the labels manually; we can just give them to the
        // hasher as-is.

        // The default 'std' hasher actually buffers 8 bytes of input before
        // processing them.  There's no point trying to chunk the input here.
        self.as_bytes()
            .iter()
            .map(|&b| b.to_ascii_lowercase())
            .for_each(|b| state.write_u8(b));
    }
}

impl AsRef<[u8]> for UncertainName {
    /// The bytes in the name in the wire format.
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl<'a> TryFrom<&'a [u8]> for &'a UncertainName {
    type Error = UncertainNameError;

    fn try_from(bytes: &'a [u8]) -> Result<Self, Self::Error> {
        UncertainName::from_bytes(bytes)
    }
}

impl<'a> From<&'a UncertainName> for &'a [u8] {
    fn from(name: &'a UncertainName) -> Self {
        name.as_bytes()
    }
}

impl<'a> From<&'a Name> for &'a UncertainName {
    fn from(value: &'a Name) -> Self {
        // SAFETY: A valid absolute name is a valid uncertain name.
        unsafe { UncertainName::from_bytes_unchecked(value.as_bytes()) }
    }
}

impl<'a> From<&'a RelName> for &'a UncertainName {
    fn from(value: &'a RelName) -> Self {
        // SAFETY: A valid relative name is a valid uncertain name.
        unsafe { UncertainName::from_bytes_unchecked(value.as_bytes()) }
    }
}

impl<'a> IntoIterator for &'a UncertainName {
    type Item = &'a Label;
    type IntoIter = Labels<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.labels()
    }
}

/// An [`UncertainName`] in a 256-byte buffer.
///
/// This is a simple wrapper around a 256-byte buffer that stores an
/// [`UncertainName`] within it.  It can be used in situations where an
/// [`UncertainName`] must be placed on the stack or within a `struct`,
/// although it is also possible to store [`UncertainName`]s on the heap as
/// `Box<UncertainName>` or `Rc<UncertainName>`.
#[derive(Clone)]
#[repr(transparent)]
pub struct UncertainNameBuf([u8; 256]);

impl UncertainNameBuf {
    /// Copy the given name.
    pub fn copy(name: &UncertainName) -> Self {
        let mut buf = [0u8; 256];
        buf[1..1 + name.len()].copy_from_slice(name.as_bytes());
        buf[0] = name.len() as u8;
        Self(buf)
    }

    /// Overwrite this by copying in a different name.
    ///
    /// Any name contained in this buffer previously will be overwritten.
    pub fn replace_with(&mut self, name: &UncertainName) {
        self.0[1..1 + name.len()].copy_from_slice(name.as_bytes());
        self.0[0] = name.len() as u8;
    }
}

impl UncertainNameBuf {
    /// The size of this name in the wire format.
    #[allow(clippy::len_without_is_empty)]
    pub const fn len(&self) -> usize {
        self.0[0] as usize
    }

    /// The wire format representation of the name.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0[1..1 + self.len()]
    }
}

impl Deref for UncertainNameBuf {
    type Target = UncertainName;

    fn deref(&self) -> &Self::Target {
        // SAFETY: 'UncertainNameBuf' always contains a valid name.
        let len = self.len();
        let bytes = &self.0[1..1 + len];
        unsafe { UncertainName::from_bytes_unchecked(bytes) }
    }
}

impl DerefMut for UncertainNameBuf {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: 'UncertainNameBuf' always contains a valid name.
        let len = self.len();
        let bytes = &mut self.0[1..1 + len];
        unsafe { UncertainName::from_bytes_unchecked_mut(bytes) }
    }
}

impl Borrow<UncertainName> for UncertainNameBuf {
    fn borrow(&self) -> &UncertainName {
        self
    }
}

impl BorrowMut<UncertainName> for UncertainNameBuf {
    fn borrow_mut(&mut self) -> &mut UncertainName {
        self
    }
}

impl AsRef<UncertainName> for UncertainNameBuf {
    fn as_ref(&self) -> &UncertainName {
        self
    }
}

impl AsMut<UncertainName> for UncertainNameBuf {
    fn as_mut(&mut self) -> &mut UncertainName {
        self
    }
}

impl From<&UncertainName> for UncertainNameBuf {
    fn from(value: &UncertainName) -> Self {
        Self::copy(value)
    }
}

/// An error in constructing an [`UncertainName`].
#[derive(Clone, Debug)]
pub struct UncertainNameError;

impl fmt::Display for UncertainNameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("could not parse an absolute/relative domain name")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for UncertainNameError {}
