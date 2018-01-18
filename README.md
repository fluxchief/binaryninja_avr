# Binary Ninja AVR plugin
This plugin adds support for the AVR architecture to binaryninja. Most of the
instructions can be lifted (mostly) correctly.

![Disassembly](https://github.com/fluxchief/binaryninja_avr/blob/master/img/disas.png "Disassembly")
![Lifted](https://github.com/fluxchief/binaryninja_avr/blob/master/img/lifted.png "Lifted")

## Installation
Run this command in your BN plugins folder:
`git clone https://github.com/fluxchief/binaryninja_avr.git`

Another option is to download this repository as a ZIP file and
extract it in your BN plugins folder.

## How is it different than [binja_avr](https://github.com/cah011/binja-avr)?
1) This project aims for a better support of different chips. It currently has

 - ATMega16
 - ATTiny48
 - ATTiny88
 - ATXMega128A4u

support and can be easily extended. Due to current restrictions of binaryninja
however, you need to hardcode the used chip in the `__init__.py`.

2) This plugin also lifts the AVR instructions. While I at first intended to add
lifting to `binja-avr`, the changes would have been to large so that I decided
to write this plugin from scratch instead.

3) Interrupt vectors are defined automatically.

4) Xrefs on memory.

## I found a bug!
"Awesome"! Please create a ticket upload your sample there as well.

## Known issues/limitations

 - No chip selection :/ Nothing we can do about at the moment (except for
   nagging the user to select an MCU at startup which is not going to happen,
   see issue #1, limitation of BN).
 - Memory accesses are weird. I had to place the data segment to an offset
   (currently 0x10 0000) because BN does not know about harvard architectures.
   This means if you have some offset in memory and want to look at this
   address, add 0x10 0000). This also causes memory access in medium IL view
   where BN could not resolve the address to look like this:
    `[GPIO0 + (123 | ((zx.w(r31) << 8) | (zx.w(r30))))].b`. GPIO0 is the first
   io register stored at RAM:0 (or 0x10 0000) - so you see where this is going.
 - Memory is treated as volatile. This makes sense for memory mapped (e)IO
   registers but we don't really want to have it for the other memory area.
   However there is nothing we can do about it, so lifting is not as good as
   it could be.
 - RAMPX/RAMPY/RAMPZ not used - This isn't lifted as much as it could be and
   decreases the readability of the code, so I disabled the use of these
   registers.
 - Skip instruction followed by a 4 bytes instruction breaks stuff. This is also
   because of a limitation of BN. BN only sends the raw bytes until the end of
   the basic block to the plugin, so there is no way we can figure out whether
   the length of the next instruction is 2 or 4 bytes. It seems to be 2 bytes
   most of the cases, so I hardcoded it to 2.
 - Flags aren't used (in lifting).

## License

[MIT](LICENSE)
