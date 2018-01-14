# Binary Ninja AVR plugin
This plugin adds support for the AVR architecture to binaryninja. Most of the
instructions can be lifted (mostly) correctly.

![Disassembly](https://github.com/fluxchief/binaryninja_avr/blob/master/img/disas.png "Disassembly")
![Lifted](https://github.com/fluxchief/binaryninja_avr/blob/master/img/lifted.png "Lifted")

## Installation
Run this command in your BN plugins folder:
`git clone https://github.com/fluxchief/binaryninja_avr.git`

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

## Known issues

 - No chip selection :/ Nothing we can do about at the moment, limitation of BN.
 - RAMPX/RAMPY/RAMPZ not used - This isn't lifted as much as it could be and
   decreases the readability of the code, so I disabled the use of these
   registers.
 - Special instructions (AES etc) - Will be added at some point, but don't
   expect proper lifting for those instructions.
 - Skip instruction followed by a 4 bytes instruction breaks stuff. This is also
   because of a limitation of BN. BN only sends the raw bytes until the end of
   the basic block to the plugin, so there is no way we can figure out whether
   the length of the next instruction is 2 or 4 bytes. It seems to be 2 bytes
   most of the cases, so I hardcoded it to 2.
 - Flags aren't used.
 - Return values > u8 aren't supported. I don't know whether this is a
   limitation of BN or if I'm just unable to implement it correctly :(.

## License

[MIT](LICENSE)
