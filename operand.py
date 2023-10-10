import abc

RAM_SEGMENT_BEGIN = 0x100000


class Operand(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, chip, value):
        self._value = value
        self._chip = chip

    # I don't know if we need both functions. We will see later. TODO.
    @abc.abstractproperty
    def immediate_value(self):
        pass

    @abc.abstractproperty
    def symbolic_value(self):
        pass

    @abc.abstractmethod
    def llil_read(self, il):
        pass

    @property
    def raw_value(self):
        return self._value


class OperandRegister(Operand):
    @property
    def immediate_value(self):
        return None

    @property
    def symbolic_value(self):
        return self._chip.registers[self._value]

    def llil_read(self, il):
        return il.reg(1, self.symbolic_value)


# Similar to OperandRegister, but consists out of two registers.
class OperandRegisterWide(Operand):
    SPECIAL_REGS = {26: 'X', 28: 'Y', 30: 'Z'}

    @property
    def immediate_value(self):
        return None

    @property
    def symbolic_value(self):
        return "{}:{}".format(
            self._chip.registers[self._value + 1],
            self._chip.registers[self._value]
        )

    def llil_read(self, il):
        if self._value in self.SPECIAL_REGS.keys():
            return il.reg(2, self.SPECIAL_REGS[self._value])

        return il.or_expr(
            2,
            il.shift_left(
                2,
                il.zero_extend(
                    2,
                    il.reg(1, self._chip.registers[self._value + 1])
                ),
                il.const(1, 8)
            ),
            il.zero_extend(2, il.reg(1, self._chip.registers[self._value])),
        )

    def low(self):
        return self._chip.registers[self._value]

    def high(self):
        return self._chip.registers[self._value + 1]


class OperandIORegister(Operand):
    @property
    def immediate_value(self):
        return None

    @property
    def symbolic_value(self):
        return self._chip.IO_REGISTERS.get(self._value, "UNKNOWN_IO_0x{:X}".format(self._value))

    def llil_read(self, il):
        return il.load(
            1,
            il.const(
                3,
                self._value + RAM_SEGMENT_BEGIN
            )
        )


class OperandImm(Operand):
    """
    Do not use, abstract immediate layer.
    """
    @property
    def immediate_value(self):
        return self._value

    @property
    def symbolic_value(self):
        return None

    def llil_read(self, il):
        return il.const(2, self.immediate_value)


class OperandDirectAddress(OperandImm):
    def llil_read(self, il):
        return il.const(3, self.immediate_value)


class OperandImmediate(OperandImm):
    pass


class OperandRelativeAddress(OperandImm):
    def llil_read(self, il):
        return il.const(3, self.immediate_value)
