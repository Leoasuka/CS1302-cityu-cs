# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% deletable=false editable=false tags=[]
# Initialization cell
try:  # for CS1302 JupyterLite pyodide kernel
    import piplite

    with open("requirements.txt") as f:
        for package in f:
            package = package.strip()
            print("Installing", package)
            await piplite.install(package)
except ModuleNotFoundError:
    pass

import jupytext
import otter
from ipywidgets import interact

grader = otter.Notebook("main.ipynb")
# %reload_ext divewidgets

# %% [markdown] slideshow={"slide_type": "slide"}
# # Lab 7: Cybersecurity

# %% [markdown] slideshow={"slide_type": "-"} tags=["remove-cell"]
# **CS1302 Introduction to Computer Programming**
# ___

# %% [markdown] slideshow={"slide_type": "subslide"}
# Python is a popular tool among hackers and engineers. In this lab, you will learn Cryptology in cybersecurity, which covers
# - [Cryptography](https://en.wikipedia.org/wiki/Cryptography): Encryption and decryption using a cipher.
# - [Cryptanalysis](https://en.wikipedia.org/wiki/Cryptanalysis): Devising an attack to break a cipher.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Caesar symmetric key cipher

# %% [markdown] slideshow={"slide_type": "fragment"}
# We first implement a simple cipher called the [Caesar cipher](https://en.wikipedia.org/wiki/Caesar_cipher).

# %% slideshow={"slide_type": "-"} tags=["hide-input"] language="html"
# <iframe width="800" height="415" src="https://www.youtube.com/embed/sMOZf4GN3oc" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

# %% [markdown] slideshow={"slide_type": "subslide"} tags=[]
# ### Encrypt/decrypt a character

# %% [markdown] slideshow={"slide_type": "fragment"}
# **How to encrypt a character?**

# %% [markdown] slideshow={"slide_type": "fragment"}
# The following code encrypts a character `char` using a non-negative integer `key`.

# %% code_folding=[] slideshow={"slide_type": "-"}
cc_n = 1114112


def cc_encrypt_character(char, key):
    """
    Return the encryption of a character by an integer key using Caesar cipher.

    Parameters
    ----------
    char: str
        a unicode (UTF-8) character to be encrypted.
    key int:
        secret key to encrypt char.
    """
    char_code = ord(char)
    shifted_char_code = (char_code + key) % cc_n
    encrypted_char = chr(shifted_char_code)
    return encrypted_char


# %% [markdown] slideshow={"slide_type": "fragment"}
# For example, to encrypt the letter `'A'` using a secret key `5`:

# %% slideshow={"slide_type": "-"}
cc_encrypt_character("A", 5)

# %% [markdown] slideshow={"slide_type": "fragment"}
# The character `'A'` is encrypted to the character `'F'` as follows:
#
# 1. `ord(char)` return the integer `65`, which is the code point (integer representation) of the unicode of `'A'`. 
# 2. `(char_code + key) % cc_n` cyclic shifts the code by the key `5`.
# 3. `chr(shifted_char_code)` converts the shifted code back to a character, which is `'F'`.
#
# | Encryption                      |     |       |     |     |     |     |     |     |
# | ------------------------------- | --- | ----- | --- | --- | --- | --- | --- | --- |
# | `char`                          | ... | **A** | B   | C   | D   | E   | F   | ... |
# | `ord(char)`                     | ... | **65**| 66  | 67  | 68  | 69  | 70  | ... |
# | `(ord(char) + key) % cc_n`      | ... | **70**| 71  | 72  | 73  | 74  | 75  | ... |
# | `(chr(ord(char) + key) % cc_n)` | ... | **F** | G   | H   | I   | J   | K   | ... |

# %% [markdown] slideshow={"slide_type": "fragment"}
# You may learn more about `ord` and `chr` from their docstrings:

# %% slideshow={"slide_type": "-"}
help(ord)
help(chr)


# %% [markdown] slideshow={"slide_type": "subslide"}
# **How to decrypt a character?**

# %% [markdown] slideshow={"slide_type": "fragment"}
# Mathematically, we define the encryption and decryption of a character for Caesar cipher as
#
# $$ \begin{aligned} E(x,k) &:= x + k \mod n & \text{(encryption)} \\
# D(x,k) &:= x - k \mod n & \text{(decryption),} \end{aligned}
# $$
# where $x$ is the character code in $\{0,\dots,n\}$ and $k$ is the secret key. `mod` operator above is the modulo operator. In Mathematics, it has a lower precedence than addition and multiplication and is typeset with an extra space accordingly.

# %% [markdown] slideshow={"slide_type": "fragment"}
# The encryption and decryption satisfy the recoverability condition
#
# $$ D(E(x,k),k) = x $$
# so two people with a common secret key can encrypt and decrypt a character, but others without the key cannot. This defines a [symmetric cipher](https://en.wikipedia.org/wiki/Symmetric-key_algorithm).

# %% [markdown] slideshow={"slide_type": "fragment"}
# The following code decrypts a character using a key.

# %% slideshow={"slide_type": "-"}
def cc_decrypt_character(char, key):
    """
    Return the decryption of a character by the key using Caesar cipher.

    Parameters
    ----------
    char: str
        a unicode (UTF-8) character to be decrypted.
    key: int
        secret key to decrypt char.
    """
    char_code = ord(char)
    shifted_char_code = (char_code - key) % cc_n
    decrypted_char = chr(shifted_char_code)
    return decrypted_char


# %% [markdown] slideshow={"slide_type": "fragment"}
# For instance, to decrypt the letter `'F'` by the secret key `5`:

# %% slideshow={"slide_type": "-"}
cc_decrypt_character("F", 5)


# %% [markdown] slideshow={"slide_type": "fragment"}
# The character `'F'` is decrypted back to `'A'` because
# `(char_code - key) % cc_n` reverse cyclic shifts the code by the key `5`.
#
# | Encryption                      |     |       |     |     |     |     |     |     | Decryption                      |
# | ------------------------------- | --- | ----- | --- | --- | --- | --- | --- | --- | ------------------------------- |
# | `char`                          | ... | **A** | B   | C   | D   | E   | F   | ... | `(chr(ord(char) - key) % cc_n)` |
# | `ord(char)`                     | ... | **65**| 66  | 67  | 68  | 69  | 70  | ... | `(ord(char) - key) % cc_n`      |
# | `(ord(char) + key) % cc_n`      | ... | **70**| 71  | 72  | 73  | 74  | 75  | ... | `ord(char)`                     |
# | `(chr(ord(char) + key) % cc_n)` | ... | **F** | G   | H   | I   | J   | K   | ... | `char`                          |

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Exercise** (modulo-arithmetic)
#
# Why did we set `cc_n = 1114112`? Explain whether the recoverability property may fail if we set `cc_n` to a bigger number or remove `% cc_n` for both `cc_encrypt_character` and `cc_decrypt_character`.

# %% [markdown]
# Because there are only 1114112 digits in unicode, so we set cc_n = 1114112 to ensure that the character's unicode is smaller than 1114112, which can be transform into character correctly. If we don't use (% cc_n) or setting cc_n to a bigger number, the (ord(char) + key) and abs(ord(char) - key) may bigger than 1114112. In this situation, the program cc_encrypt_character and cc_decrypt_character may raise ValueError so both of them will fail.

# %% [markdown] slideshow={"slide_type": "slide"} tags=[]
# ### Encrypt a plaintext and decrypt a ciphertext

# %% [markdown] slideshow={"slide_type": "fragment"}
# Of course, it is more interesting to encrypt a string instead of a character. The following code implements this in one line.

# %% slideshow={"slide_type": "-"}
def cc_encrypt(plaintext, key):
    """
    Return the ciphertext of a plaintext by the key using the Caesar cipher.

    Parameters
    ----------
    plaintext: str
        A unicode (UTF-8) message to be encrypted.
    public_key: int
        Public key to encrypt plaintext.
    """
    return "".join([chr((ord(char) + key) % cc_n) for char in plaintext])


# %% [markdown] slideshow={"slide_type": "fragment"}
# The above function encrypts a message, referred to as the *plaintext*, by replacing each character with its encryption.  
# This is referred to as a [*substitution cipher*](https://en.wikipedia.org/wiki/Substitution_cipher).

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Exercise** (cc-decrypt)
#
# Define a function `cc_decrypt` that
# - takes a string `ciphertext` and an integer `key`, and
# - returns the plaintext that encrypts to `ciphertext` by the key using Caesar cipher.

# %% nbgrader={"grade": false, "grade_id": "cc_decrypt", "locked": false, "schema_version": 3, "solution": true, "task": false} slideshow={"slide_type": "-"} tags=["remove-output"]
def cc_decrypt(ciphertext, key):
    """
    Return the plaintext that encrypts to ciphertext by the key using Caesar cipher.

    Parameters
    ----------
    ciphertext: str
        message to be decrypted.
    key: int
        secret key to decrypt the ciphertext.
    """
    return ''.join([cc_decrypt_character(char, key) for char in ciphertext])


# %% deletable=false editable=false
grader.check("cc-decrypt")

# %% [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Brute-force attack

# %% [markdown]
# ### Create an English dictionary

# %% [markdown] slideshow={"slide_type": "fragment"}
# You will launch a brute-force attack to guess the key that encrypts an English text. The idea is simple: 
#
# - You try decrypting the ciphertext with different keys, and 
# - see which of the resulting plaintexts make the most sense (most English-like).

# %% [markdown] slideshow={"slide_type": "fragment"}
# To check whether a plaintext is English-like, we need to have a list of English words. One way is to type them out
# but this is tedious. Alternatively, we can obtain the list from the *Natural Language Toolkit (NLTK)*:

# %%
# !pip install nltk

# %% slideshow={"slide_type": "-"}
import nltk

nltk.download("words")
from nltk.corpus import words

# %% [markdown] slideshow={"slide_type": "subslide"}
# `words.words()` returns a list of words. We can check whether a string is in the list using the operator `in`.

# %% slideshow={"slide_type": "-"}
for word in "Ada", "ada", "Hello", "hello":
    print("{!r} in dictionary? {}".format(word, word in words.words()))

# %% [markdown] slideshow={"slide_type": "fragment"}
# However, there are two issues:
# - Checking membership is slow for a long list.
# - Both 'Hello' and 'ada' are English-like but not in the words list.

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Exercise** (dictionary)
#
# Using the method `lower` of `str` and the constructor `set`, assign `dictionary` to a set of lowercase English words from `words.words()`.

# %% nbgrader={"grade": false, "grade_id": "nltk", "locked": false, "schema_version": 3, "solution": true, "task": false} slideshow={"slide_type": "-"} tags=["remove-output"]
dictionary = set(word.lower() for word in words.words()) 

# %% deletable=false editable=false
grader.check("dictionary")


# %% [markdown] slideshow={"slide_type": "subslide"} tags=[]
# ### Identify English-like text

# %% [markdown] slideshow={"slide_type": "fragment"}
# To determine how English-like a text is, we calculate the following score:
#
# $$
# \frac{\text{number of English words in the text}}{\text{number of tokens in the text}} 
# $$
# where tokens are substrings, not necessarily English words, separated by white space characters in the text.

# %% slideshow={"slide_type": "-"}
def tokenizer(text):
    """Returns the list of tokens of the text."""
    return text.split()


def get_score(text):
    """Returns the fraction of tokens which appear in dictionary."""
    tokens = tokenizer(text)
    words = [token for token in tokens if token in dictionary]
    return len(words) / len(tokens)


# tests
get_score("hello world"), get_score("Hello, World!")

# %% [markdown] slideshow={"slide_type": "fragment"}
# As shown in the tests above, the code fails to handle text with punctuations and uppercase letters properly.  
# In particular, 
# - while `get_score` recognizes `hello world` as English-like and returns the maximum score of 1, 
# - it fails to recognize `Hello, World!` as English-like and returns the minimum score of 0.

# %% [markdown] slideshow={"slide_type": "fragment"}
# Why? Every word in `dictionary`
# - are in lowercase, and
# - have no leading/trailing punctuations.

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Exercise** (tokenizer)
#
# Define a function `tokenizer` that 
# - takes a string `text` as an argument, and
# - returns a `list` of tokens obtained by
#   1. splitting `text` into a list using `split()`;
#   2. removing leading/trailing punctuations in `string.punctuation` using the `strip` method; and
#   3. converting all items of the list to lowercase using `lower()`.

# %% nbgrader={"grade": false, "grade_id": "tokenizer", "locked": false, "schema_version": 3, "solution": true, "task": false} slideshow={"slide_type": "-"} tags=["remove-output"]
import string


def tokenizer(text):
    """Returns the list of tokens of the text such that
    1) each token has no leading or trailing spaces/punctuations, and
    2) all letters in each token are in lowercase."""
    origin_text = text.split()
    ans = []
    punctuation = "',.?!"
    pun = '"'
    for word in origin_text:
        trans_word = ''
        for cha in word.lower():
            if cha not in punctuation and cha not in pun:
                trans_word += cha
        ans.append(trans_word)
    return ans
            


# %% deletable=false editable=false
grader.check("tokenizer")


# %% [markdown] slideshow={"slide_type": "slide"} tags=[]
# ### Launch a brute-force attack

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Exercise** (cc-attack)
#
# Define the function `cc_attack` that 
# - takes as arguments
#     - a string `ciphertext`,
#     - a floating point number `threshold` in the interval $(0,1)$ with a default value of $0.6$, and
# - returns a generator that  
#     - generates one-by-one in ascending order guesses of the key that
#     - decrypt `ciphertext` to texts with scores at least the `threshold`.

# %% nbgrader={"grade": false, "grade_id": "cc_attack", "locked": false, "schema_version": 3, "solution": true, "task": false} slideshow={"slide_type": "-"} tags=["remove-output"]
def cc_attack(ciphertext, threshold=0.6):
    """Returns a generator that generates the next guess of the key that
    decrypts the ciphertext to a text with get_score(text) at least the threshold.
    """
    key = 0
    while key >= 0:
        if get_score(cc_decrypt(ciphertext, key)) > threshold:
            yield key
        else:
            pass
        key += 1


# %% deletable=false editable=false
grader.check("cc-attack")


# %% [markdown] slideshow={"slide_type": "slide"}
# ## Challenge

# %% [markdown]
# Another symmetric key cipher is [columnar transposition cipher](https://en.wikipedia.org/wiki/Transposition_cipher#Columnar_transposition). A transposition cipher encrypts a text by permuting instead of substituting characters.

# %% [markdown] slideshow={"slide_type": "subslide"}
# **Exercise** (columnar)
#
# Study and implement the irregular case of the [columnar transposition cipher](https://en.wikipedia.org/wiki/Transposition_cipher#Columnar_transposition) as described in the Wikipedia page. Define the functions 
# - `ct_encrypt(plaintext, key)` for encryption, and 
# - `ct_decrypt(ciphertext, key)` for decryption. 
#
# You can assume the plaintext is in uppercase and has no spaces/punctuations.

# %% [markdown]
# ```{hint}
# See the test cases for examples of `plaintext`, `key`, and the corresponding `ciphertext`.
# ```

# %% nbgrader={"grade": false, "grade_id": "ct", "locked": false, "schema_version": 3, "solution": true, "task": false} slideshow={"slide_type": "-"} tags=["remove-output"]
def argsort(seq):
    '''A helper function that returns the tuple of indices that would sort the
    sequence seq.'''
    return tuple(x[0] for x in sorted(enumerate(seq), key=lambda x: x[1]))


#def ct_idx(length, key):
 #   '''A helper function that returns the tuple of indices that would permute 
  #  the letters of a message according to the key using the irregular case of 
   # columnar transposition cipher.'''
    #seq = tuple(range(length))
    #return [i for j in argsort(key) ]
def ct_idx(length, key):
    '''A helper function that returns the tuple of indices that would permute 
    the letters of a message according to the key using the irregular case of 
    columnar transposition cipher.'''
    seq = tuple(range(length))
    #return [i for j in argsort(key) for i in 
    sort_lis = []
    for j in argsort(key):
        n = 0       
        while j + n*len(key) < length:
            sort_lis.append(j+n*len(key))
            n += 1
    return sort_lis


def ct_encrypt(plaintext, key):
    '''
    Return the ciphertext of a plaintext by the key using the irregular case
    of columnar transposition cipher.

    Parameters
    ----------
    plaintext: str
        a message in uppercase without punctuations/spaces.
    key: str
        secret key to encrypt plaintext.
    '''
    return ''.join([plaintext[i] for i in ct_idx(len(plaintext), key)])


def ct_decrypt(ciphertext, key):
    '''
    Return the plaintext of the ciphertext by the key using the irregular case
    of columnar transposition cipher.

    Parameters
    ----------
    ciphertext: str
        a string in uppercase without punctuations/spaces.
    key: str
        secret key to decrypt ciphertext.
    '''
    key_sort = ct_idx(len(ciphertext), key)
    ans_dic = {}
    for i in range(len(ciphertext)):
        ans_dic[key_sort[i]] = ciphertext[i]
    return ''.join([ans_dic[i] for i in range(len(ciphertext))])


# %% deletable=false editable=false
grader.check("columnar")

# %% [markdown] editable=false tags=["remove-cell"]
# ## Submission
#
# Make sure you have run all cells in your notebook in order before running the cell below, so that all images/graphs appear in the output. The cell below will generate a zip file for you to submit. **Please save before exporting!**

# %% deletable=false editable=true tags=[]
# extra files to submit
extra_files = []

# %% deletable=false editable=false tags=[]
# Generate the source main.py necessary for grading and similarity check.
jupytext.write(jupytext.read("main.ipynb"), "main.py", fmt="py:percent")

# %% deletable=false editable=false tags=[]
# Generate the zip file to submit.
grader.export(pdf=False, run_tests=False, files=["main.py",  *extra_files])

# %%
grader.export(pdf=False, run_tests=False, files=["main.py"])

# %%
