# blocknet-vanity

Vanity address generator for Blocknet. Brute-forces keypairs to find addresses matching a given prefix and/or suffix.

## how it works

A Blocknet address is `base58(spend_public_key || view_public_key)`. The generator fixes the view keypair per thread and increments the spend scalar, requiring only one scalar multiplication per candidate. Matching is case-insensitive.

## download

Pre-built binaries at [blocknetcrypto.com](https://blocknetcrypto.com). Single binary, no extra files needed.

## build from source

Requires Rust 1.75+.

### linux

```
sudo apt install build-essential
make
```

### macos

```
xcode-select --install
make
```

### windows (msys2)

Install MSYS2 from https://www.msys2.org/, then in MINGW64 shell:

```
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-rust
make
```

Or without make:

```
cargo build --release
```

## run

```
./blocknet-vanity --prefix abc
```

### flags

```
--prefix <str>      prefix the address must start with (case-insensitive)
--suffix <str>      suffix the address must end with (case-insensitive)
-t, --threads <N>   number of worker threads (default: CPU core count)
```

At least one of `--prefix` or `--suffix` is required. Both can be combined. Patterns are limited to 8 characters and must only contain valid base58 characters (no `0`, `O`, `I`, `l`).

### examples

```
./blocknet-vanity --prefix dead
./blocknet-vanity --suffix cat
./blocknet-vanity --prefix ab --suffix cd -t 8
```

### output

Found wallets are saved in ./wallets/{pattern}-{prefix|suffix}/{pattern}.json

```json
{
  "address": "deadB7x8...",
  "spend_private_key": "...",
  "spend_public_key": "...",
  "view_private_key": "...",
  "view_public_key": "..."
}
```

## license

BSD 3-Clause. See LICENSE.
