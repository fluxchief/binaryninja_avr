import abc
import binaryninja


# Hack to have abstract static methods with abc
class abstractstatic(staticmethod):
    __slots__ = ()

    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True
    __isabstractmethod__ = True


class Chip(object):
    __metaclass__ = abc.ABCMeta
    """
    Different chips have a different amount of pins, interrupt handlers etc.

    The data can be extracted from the corresponding avr header files (e.g.
    /usr/avr/include/avr/iom16.h for atmega16).
    """
    # Names of the chip.
    CHIP_ALIASES = []
    # Size of the RAM/ROM.
    RAM_SIZE = 32
    ROM_SIZE = 128
    # The actual ram starts behind the extended IO registers, use a sane default
    # value here (that matches most of the chips).
    RAM_STARTS_AT = 0x100
    # The MCU maps R0-R31 to the beginning of its memory.
    ARCHITECTURE_MAPS_REGISTERS = True
    # Size of the interrupt vector entries (2 or 4, `_VECTOR_SIZE`).
    INTERRUPT_VECTOR_SIZE = 2
    # List of the interrupt vectors.
    INTERRUPT_VECTORS = []
    # Regular and extended IO registers (dict(offset -> name)).
    IO_REGISTERS = {}
    EXTENDED_IO_REGISTERS = {}

    def __init__(self):
        # Caching stuff.
        self.__reg_name_to_offset = {
            v: k for k, v in self.all_registers.items()
        }

    @property
    def all_registers(self):
        all_regs = dict()

        io_offset = 0
        if (self.ARCHITECTURE_MAPS_REGISTERS):
            all_regs.update(self.registers)
            io_offset = 0x20

        all_regs.update({
            k + io_offset: v
            for k, v in self.IO_REGISTERS.items()
        })
        all_regs.update({
            k + 0x00: v
            for k, v in self.EXTENDED_IO_REGISTERS.items()
        })

        return all_regs

    @property
    def registers(self):
        """
        Probably all AVR chips have the same registers
        """
        return {
            i: "r{}".format(i) for i in range(32)
        }

    def get_register_offset(self, reg_name):
        if reg_name not in self.__reg_name_to_offset:
            binaryninja.log.log_error("Invalid register: {}".format(reg_name))

        return self.__reg_name_to_offset.get(reg_name, 0xBAD0)
