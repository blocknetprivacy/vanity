BINARY = blocknet-vanity

all: build

build:
	cargo build --release
	cp target/release/$(BINARY) .

clean:
	cargo clean
	rm -f $(BINARY)

.PHONY: all build clean
