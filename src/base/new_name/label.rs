use core::{
    borrow::{Borrow, BorrowMut},
    cmp, fmt,
    hash::{Hash, Hasher},
    iter,
    ops::{Deref, DerefMut},
};

use super::UncertainName;

/// A label in a domain name.
#[repr(transparent)]
pub struct Label([u8]);

impl Label {
    /// The maximum size of a label in the wire format.
    pub const MAX_SIZE: usize = 63;

    /// The root label.
    pub const ROOT: &'static Self =
        unsafe { Self::from_bytes_unchecked(b"") };

    /// The wildcard label.
    pub const WILDCARD: &'static Self =
        unsafe { Self::from_bytes_unchecked(b"*") };
}

impl Label {
    /// Assume a byte string is a valid [`Label`].
    ///
    /// # Safety
    ///
    /// The byte string must be within the size restriction (63 bytes or
    /// fewer).
    pub const unsafe fn from_bytes_unchecked(bytes: &[u8]) -> &Self {
        // SAFETY: 'Label' is a 'repr(transparent)' wrapper around '[u8]', so
        // casting a '[u8]' into a 'Label' is sound.
        core::mem::transmute(bytes)
    }

    /// Assume a mutable byte string is a valid [`Label`].
    ///
    /// # Safety
    ///
    /// The byte string must be within the size restriction (63 bytes or
    /// fewer).
    pub unsafe fn from_bytes_unchecked_mut(bytes: &mut [u8]) -> &mut Self {
        // SAFETY: 'Label' is a 'repr(transparent)' wrapper around '[u8]', so
        // casting a '[u8]' into a 'Label' is sound.
        core::mem::transmute(bytes)
    }

    /// Try converting a byte string into a [`Label`].
    ///
    /// If the byte string is too long, an error is returned.
    ///
    /// Runtime: `O(bytes.len())`.
    pub fn from_bytes(bytes: &[u8]) -> Result<&Self, LabelError> {
        if bytes.len() > Self::MAX_SIZE {
            // The label was too long to be used.
            return Err(LabelError);
        }

        Ok(unsafe { Self::from_bytes_unchecked(bytes) })
    }

    /// Extract a label from the start of a byte string.
    ///
    /// A label encoded in the wire format will be extracted from the beginning
    /// of the given byte string.  If a valid label cannot be extracted, or the
    /// byte string is simply empty, an error is returned.  The extracted label
    /// and the remainder of the byte string are returned.
    ///
    /// Runtime: `O(1)`.
    pub fn split_off(bytes: &[u8]) -> Result<(&Self, &[u8]), LabelError> {
        let (&length, bytes) = bytes.split_first().ok_or(LabelError)?;
        if length < 64 && bytes.len() >= length as usize {
            let (label, bytes) = bytes.split_at(length as usize);
            // SAFETY: 'label' is known be to less than 64 bytes in size.
            Ok((unsafe { Self::from_bytes_unchecked(label) }, bytes))
        } else {
            // Overlong label (or compression pointer).
            Err(LabelError)
        }
    }
}

impl Label {
    /// Whether this is the root label.
    pub const fn is_root(&self) -> bool {
        self.0.is_empty()
    }

    /// Whether this is the wildcard label.
    pub const fn is_wildcard(&self) -> bool {
        // NOTE: 'self.0 == *b"*"' is not const.
        self.0.len() == 1 && self.0[0] == b'*'
    }

    /// The size of this name in the wire format.
    #[allow(clippy::len_without_is_empty)]
    pub const fn len(&self) -> usize {
        self.0.len()
    }

    /// The wire format representation of the name.
    pub const fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

impl Label {
    /// Whether this is an LDH label.
    ///
    /// LDH ("letter-digit-hyphen") labels consist exclusively of ASCII letter
    /// (A-Z, a-z), digit (0-9), and hyphen (-) characters, where labels begin
    /// and end with non-hyphen characters.
    ///
    /// See [RFC 5890, section 2.3.1].  This is also known as the "preferred
    /// name syntax" of [RFC 1034, section 3.5].
    ///
    /// [RFC 5890, section 2.3.1]: https://datatracker.ietf.org/doc/html/rfc5890#section-2.3.1
    /// [RFC 1034, section 3.5]: https://datatracker.ietf.org/doc/html/rfc1034#section-3.5
    pub fn is_ldh(&self) -> bool {
        self.as_bytes()
            .iter()
            .all(|&b| b.is_ascii_alphanumeric() || b == b'-')
            && !self.as_bytes().starts_with(b"-")
            && !self.as_bytes().ends_with(b"-")
    }

    /// Whether this is an NR-LDH label.
    ///
    /// A "non-reserved" LDH label is slightly stricter than an LDH label (see
    /// [`Self::is_ldh()`]); it further does not allow the third and fourth
    /// characters to both be hyphens.  A-labels (Unicode labels encoded into
    /// ASCII) are not NR-LDH labels as they begin with `xn--`.
    ///
    /// See [RFC 5890, section 2.3.1].
    ///
    /// [RFC 5890, section 2.3.1]: https://datatracker.ietf.org/doc/html/rfc5890#section-2.3.1
    pub fn is_nr_ldh(&self) -> bool {
        self.is_ldh() && self.as_bytes().get(2..4) != Some(b"--")
    }
}

impl Label {
    /// Canonicalize this label.
    ///
    /// All uppercase ASCII characters in the label will be lowercased.
    ///
    /// Runtime: `O(self.len())`.
    pub fn canonicalize(&mut self) {
        self.0.make_ascii_lowercase()
    }
}

impl PartialEq for Label {
    /// Compare labels by their canonical value.
    ///
    /// Canonicalized labels have uppercase ASCII characters lowercased, so this
    /// function compares the two names ASCII-case-insensitively.
    ///
    // Runtime: `O(self.len())`, which is equal to `O(that.len())`.
    fn eq(&self, that: &Self) -> bool {
        self.0.eq_ignore_ascii_case(&that.0)
    }
}

impl Eq for Label {}

impl PartialOrd for Label {
    /// Compare labels by their canonical value.
    ///
    /// Canonicalized labels have uppercase ASCII characters lowercased, so this
    /// function compares the two names ASCII-case-insensitively.
    ///
    // Runtime: `O(self.len())`, which is equal to `O(that.len())`.
    fn partial_cmp(&self, that: &Self) -> Option<cmp::Ordering> {
        Some(Ord::cmp(self, that))
    }
}

impl Ord for Label {
    /// Compare labels by their canonical value.
    ///
    /// Canonicalized labels have uppercase ASCII characters lowercased, so this
    /// function compares the two names ASCII-case-insensitively.
    ///
    // Runtime: `O(self.len())`, which is equal to `O(that.len())`.
    fn cmp(&self, that: &Self) -> cmp::Ordering {
        let this_bytes = self.as_bytes().iter().copied();
        let that_bytes = that.as_bytes().iter().copied();
        iter::zip(this_bytes, that_bytes)
            .find(|(l, r)| !l.eq_ignore_ascii_case(r))
            .map_or(Ord::cmp(&self.len(), &that.len()), |(l, r)| {
                Ord::cmp(&l.to_ascii_lowercase(), &r.to_ascii_lowercase())
            })
    }
}

impl Hash for Label {
    /// Hash this label by its canonical value.
    ///
    /// The hasher is provided with the labels in this name with ASCII
    /// characters lowercased.  Each label is preceded by its length as `u8`.
    ///
    /// The same scheme is used by [`Name`] and [`RelName`], so a tuple of any
    /// of these types will have the same hash as the concatenation of the
    /// labels.
    ///
    /// [`Name`]: super::Name
    /// [`RelName`]: super::RelName
    ///
    /// Runtime: `O(self.len())`.
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Individual labels and names should hash in the same way.
        state.write_u8(self.len() as u8);

        // The default 'std' hasher actually buffers 8 bytes of input before
        // processing them.  There's no point trying to chunk the input here.
        for &b in self.as_bytes() {
            state.write_u8(b.to_ascii_lowercase());
        }
    }
}

impl AsRef<[u8]> for Label {
    /// The raw bytes in this name, with no length octet.
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl<'a> TryFrom<&'a [u8]> for &'a Label {
    type Error = LabelError;

    fn try_from(bytes: &'a [u8]) -> Result<Self, Self::Error> {
        Label::from_bytes(bytes)
    }
}

impl<'a> From<&'a Label> for &'a [u8] {
    fn from(label: &'a Label) -> Self {
        label.as_bytes()
    }
}

/// A [`Label`] in a 64-byte buffer.
///
/// This is a simple wrapper around a 64-byte buffer that stores a [`Label`]
/// within it.  It can be used in situations where a [`Label`] must be placed
/// on the stack or within a `struct`, although it is also possible to store
/// [`Label`]s on the heap as `Box<Label>` or `Rc<Label>`.
#[derive(Clone)]
#[repr(transparent)]
pub struct LabelBuf([u8; 64]);

impl LabelBuf {
    /// Copy the given label.
    pub fn copy(label: &Label) -> Self {
        let mut buf = [0u8; 64];
        buf[1..1 + label.len()].copy_from_slice(label.as_bytes());
        buf[0] = label.len() as u8;
        Self(buf)
    }

    /// Overwrite this by copying in a different label.
    ///
    /// Any label contained in this buffer previously will be overwritten.
    pub fn replace_with(&mut self, label: &Label) {
        self.0[1..1 + label.len()].copy_from_slice(label.as_bytes());
        self.0[0] = label.len() as u8;
    }
}

impl LabelBuf {
    /// The size of this label, without the length octet.
    #[allow(clippy::len_without_is_empty)]
    pub const fn len(&self) -> usize {
        self.0[0] as usize
    }

    /// The bytes in the label, without the length octet.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0[1..1 + self.len()]
    }

    /// Treat this as an uncertain name.
    pub fn as_name(&self) -> &UncertainName {
        // SAFETY: A valid label with length octet is a valid name.
        unsafe { UncertainName::from_bytes_unchecked(&self.0) }
    }
}

impl Deref for LabelBuf {
    type Target = Label;

    fn deref(&self) -> &Self::Target {
        // SAFETY: 'LabelBuf' always contains a valid label.
        let len = self.len();
        let bytes = &self.0[1..1 + len];
        unsafe { Label::from_bytes_unchecked(bytes) }
    }
}

impl DerefMut for LabelBuf {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: 'LabelBuf' always contains a valid label.
        let len = self.len();
        let bytes = &mut self.0[1..1 + len];
        unsafe { Label::from_bytes_unchecked_mut(bytes) }
    }
}

impl Borrow<Label> for LabelBuf {
    fn borrow(&self) -> &Label {
        self
    }
}

impl BorrowMut<Label> for LabelBuf {
    fn borrow_mut(&mut self) -> &mut Label {
        self
    }
}

impl AsRef<Label> for LabelBuf {
    fn as_ref(&self) -> &Label {
        self
    }
}

impl AsMut<Label> for LabelBuf {
    fn as_mut(&mut self) -> &mut Label {
        self
    }
}

impl From<&Label> for LabelBuf {
    fn from(value: &Label) -> Self {
        Self::copy(value)
    }
}

/// An error in constructing a [`Label`].
#[derive(Clone, Debug)]
pub struct LabelError;

impl fmt::Display for LabelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("could not parse a domain name label")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for LabelError {}
