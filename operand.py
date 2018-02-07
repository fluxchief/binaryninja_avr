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
    @property
    def immediate_value(self):
        return None

    @property
    def symbolic_value(self):
        return "{}:{}".format(
            self._chip.registers[self._value],
            self._chip.registers[self._value + 1]
        )

    def llil_read(self, il):
        return il.or_expr(
            2,
            il.shift_left(
                2,
                il.zero_extend(2,
                    il.reg(1, self._chip.registers[self._value + 1])
                ),
                il.const(1, 8)
            ),
            il.zero_extend(2, il.reg(1, self._chip.registers[self._value])),
        )


class OperandIORegister(Operand):
    @property
    def immediate_value(self):
        return None

    @property
    def symbolic_value(self):
        return self._chip.IO_REGISTERS[self._value]

    def llil_read(self, il):
        return il.load(
            1,
            il.const(
                3,
                self._chip.get_register_offset(
                    self.symbolic_value) + RAM_SEGMENT_BEGIN
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
