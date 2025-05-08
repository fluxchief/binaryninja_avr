import abc
import collections
import struct

from . import operand
from .operand import RAM_SEGMENT_BEGIN

try:
    import binaryninja
    from binaryninja.log import log
    from binaryninja.enums import LogLevel
    from binaryninja import InstructionTextToken, InstructionTextTokenType, LowLevelILFlagCondition
except Exception:
    class LogLevel:
        AlertLog = 'ALERT'
        DebugLog = 'DEBUG'
        ErrorLog = 'ERROR'
        InfoLog = 'INFO'
        WarningLog = 'WARNING'

    def log(level, text):
        print("[{}]: {}".format(level, text))


# Hack to have abstract static methods with abc
class abstractstatic(staticmethod):
    __slots__ = ()

    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True
    __isabstractmethod__ = True


class Instruction(object):
    __metaclass__ = abc.ABCMeta

    # The order in which the operands have in the instruction. Will only be honored
    # as long as args_to_operand is not overwritten
    register_order = []

    def __init__(self, chip, addr, data, operands):
        # Sanity check
        assert len(operands) == self.num_operands(), \
            "{}: Expected {} operands, got {}".format(
                self.__class__.__name__,
                len(operands),
                self.num_operands())
        self._chip = chip
        self._addr = addr
        self._data = data
        self._operands = operands

    @classmethod
    def name(cls):
        """
        Should be fine for most instructions, multiple implementations of the
        same instruction however should override this function
        """
        try:
            return cls.__name__.split("_")[1].lower()
        except Exception:
            return cls.__name__

    @abstractstatic
    def instruction_signature():
        """
        From the manual, e.g. '0010 00rd dddd rrrr'.
        Will be used to generate the parts of the class automatically ;)

        Valid chars:
            '0', '1',
            'r', 'd',
            'q', (displacement)
        """

    @staticmethod
    def verify_args(args):
        return True

    @staticmethod
    def _arg_to_operand(chip, arg_type, arg):
        if arg_type in ('r', 'd'):
            return operand.OperandRegister(chip, arg)
        elif arg_type in ('R', 'D'):
            return operand.OperandRegisterWide(chip, arg)
        elif arg_type in ('k', 'K', 'q', 'b'):
            return operand.OperandImmediate(chip, arg)
        elif arg_type == 'A':
            return operand.OperandIORegister(chip, arg)
        else:
            log(LogLevel.WarningLog, "{} not correctly supported yet".format(arg_type))
            return operand.OperandImmediate(chip, arg)

    @classmethod
    def args_to_operands(cls, chip, args):
        """
        Autodetect register types. Might work for some, will work out bad for others.
        """
        operands = []
        # If we have the order in which the registers should appear, use it
        if cls.register_order:
            for order_i in cls.register_order:
                operands.append(Instruction._arg_to_operand(
                    chip, order_i, args[order_i]
                ))

            return operands

        # fall back to some generic algorithm
        for k, arg in args.items():
            operands.append(Instruction._arg_to_operand(chip, k, arg))

        return operands

    @classmethod
    def parse_value(cls, chip, addr, data, instruction):
        """
        Tries to parse the value and returns a cls object if possible, otherwise returns None
        """

        # Pattern match
        pattern = cls.instruction_signature()

        pattern_matches = collections.defaultdict(lambda: [])

        for i in range(len(pattern)):
            instr_i = len(pattern) - i - 1
            bit = (instruction >> instr_i) & 1
            if pattern[i] in ['0', '1']:
                # Those must match, otherwise it's not this opcode
                if int(pattern[i]) != bit:
                    return None
            # elif pattern[i] in ['r', 'R', 'd', 'D', 'q', 'k', 'K', 'A']:
            else:
                pattern_matches[pattern[i]].append(bit)

        # Convert list of bits to actual values
        for k in pattern_matches.keys():
            pattern_matches[k] = int(''.join([str(_) for _ in pattern_matches[k]]), 2)

        # Check if the arguments are in valid ranges, then convert them to the actual
        # operands objects
        if not cls.verify_args(pattern_matches):
            return None

        args = cls.args_to_operands(chip, pattern_matches)

        # Autodetect the types of the argument
        return cls(chip, addr, data, args)

    @classmethod
    def num_operands(cls):
        """
        Number of operands for this instruction
        """
        nargs = 0
        lsig = cls.instruction_signature()

        if 'r' in lsig.lower():
            nargs += 1

        if 'd' in lsig.lower():
            nargs += 1

        if 'k' in lsig.lower():
            nargs += 1

        if 'a' in lsig.lower():
            nargs += 1

        if 'b' in lsig.lower():
            nargs += 1

        if 's' in lsig.lower():
            nargs += 1

        if 'q' in lsig.lower():
            nargs += 1

        return nargs

    @property
    def operands(self):
        return self._operands

    @classmethod
    def length(cls):
        """
        Length of this instruction. In most cases this should be 2.
        """
        return len(cls.instruction_signature()) // 8

    def get_instruction_text(self):
        name = self.name()

        tokens = [
            InstructionTextToken(InstructionTextTokenType.InstructionToken, name)
        ]

        if self.num_operands():
            # No idea if I should use the operand separator token here as well.
            tokens.append(InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ' '
            ))

        for idx, op in enumerate(self.operands):
            if (type(op) in [operand.OperandRegister,
                             operand.OperandRegisterWide,
                             operand.OperandIORegister]):
                op_token = InstructionTextToken(
                    InstructionTextTokenType.RegisterToken,
                    op.symbolic_value
                )

            elif (type(op) in [operand.OperandDirectAddress,
                               operand.OperandRelativeAddress]):
                v = op.immediate_value
                if isinstance(op, operand.OperandRelativeAddress):
                    v += self._addr
                    if v >= self._chip.ROM_SIZE:
                        v -= self._chip.ROM_SIZE
                    if v < 0:
                        v += self._chip.ROM_SIZE

                op_token = InstructionTextToken(
                    InstructionTextTokenType.PossibleAddressToken,
                    hex(int(v)),
                    value=v
                )
            elif type(op) is operand.OperandImmediate:
                op_token = InstructionTextToken(
                    InstructionTextTokenType.IntegerToken,
                    hex(int(op.immediate_value)),
                    value=op.immediate_value
                )
            else:
                raise RuntimeError(
                    "Unhandled op type: {}".format(op.__class__.__name__)
                )

            tokens.append(op_token)
            if idx < self.num_operands() - 1:
                tokens.append(InstructionTextToken(
                    InstructionTextTokenType.OperandSeparatorToken,
                    ', '
                ))

        return tokens

    def get_llil(self, il):
        """
        Translates the instruction to LLIL instructions.
        """
        il.append(il.unimplemented())

# Returns RAMP* register if existing, otherwise zero.
def get_ramp_register(il, chip, ramp):
    if ramp in chip.all_registers.values():
        return il.load(
            1,
            il.const(
                3,
                RAM_SEGMENT_BEGIN + chip.get_register_offset(ramp)
            )
        )
    else:
        return il.const(1, 0)

def get_reg_pair(il, rhigh, rlow):
    repls = {'r26': 'X', 'r28': 'Y', 'r30': 'Z'}
    if repls.get(rlow.lower()):
        return il.reg(2, repls[rlow.lower()])
    else:
        return il.or_expr(
            2,
            il.shift_left(
                2,
                il.reg(1, rhigh),
                il.const(1, 8)
            ),
            il.zero_extend(2, il.reg(1, rlow))
        )

def get_xyz_register(il, chip, r, do_ramp=True):
    r = r.upper()
    assert r in ['X', 'Y', 'Z']
    repls = {'X': ('r26', 'r27'), 'Y': ('r28', 'r29'), 'Z': ('r30', 'r31')}
    # TODO: We 'support' RAMPX..RAMZ, however BN is unable to handle it in a way
    # that an access to a reg is shown:
    #   [REG].b = X vs [GPIO0 + (([RAMPZ].b << 0x10) | addrof(X))])
    # This is why we disable it here.
    if False and do_ramp:
        # Use RAMPX/Y/Z register
        return il.or_expr(
            3,
            il.shift_left(
                3,
                get_ramp_register(il, chip, 'RAMP' + r),
                il.const(1, 16)
            ),
            il.reg(2, r)
        )
    else:
        return il.reg(2, r)

# Definitions of the instructions from the
# Atmel-0856-AVR-Instruction-Set-Manual.


class Instruction_ADC(Instruction):
    register_order = ['d', 'r']

    @staticmethod
    def instruction_signature():
        return '0001 11rd dddd rrrr'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.add_carry(
                    1,
                    self._operands[0].llil_read(il),
                    self._operands[1].llil_read(il),
                    il.flag('C'),
                    'HSVNZC'
                )
            )
        )


class Instruction_ADD(Instruction):
    register_order = ['d', 'r']

    @staticmethod
    def instruction_signature():
        return '0000 11rd dddd rrrr'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.add(
                    1,
                    self._operands[0].llil_read(il),
                    self._operands[1].llil_read(il),
                    'HSVNZC'
                )
            )
        )


class Instruction_ADIW(Instruction):
    @classmethod
    def args_to_operands(cls, chip, args):
        k = args['K']
        d = args['d']

        return [
            operand.OperandRegisterWide(chip, d * 2 + 24),
            operand.OperandImmediate(chip, k)
        ]

    @staticmethod
    def instruction_signature():
        return '1001 0110 KKdd KKKK'.replace(' ', '')

    def get_llil(self, il):
        rhigh, rlow = self._operands[0].symbolic_value.split(':')
        il.append(
            il.set_reg_split(
                1,
                rhigh, rlow,
                il.add(
                    2,
                    get_reg_pair(il, rhigh, rlow),
                    self._operands[1].llil_read(il),
                    'SVNZC'
                )
            )
        )


class Instruction_AND(Instruction):
    register_order = ['d', 'r']

    @staticmethod
    def instruction_signature():
        return '0010 00rd dddd rrrr'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.and_expr(
                    1,
                    self._operands[0].llil_read(il),
                    self._operands[1].llil_read(il),
                    'SVNZ'
                )
            )
        )


class Instruction_ANDI(Instruction):
    @classmethod
    def args_to_operands(cls, chip, args):
        k = args['K']
        d = args['d']

        return [
            operand.OperandRegister(chip, d + 16),
            operand.OperandImmediate(chip, k)
        ]

    @staticmethod
    def instruction_signature():
        return '0111 KKKK dddd KKKK'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.and_expr(
                    1,
                    self._operands[0].llil_read(il),
                    self._operands[1].llil_read(il),
                    'SVNZ'
                )
            )
        )


class Instruction_ASR(Instruction):
    @staticmethod
    def instruction_signature():
        return '1001 010d dddd 0101'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.arith_shift_right(
                    1,
                    self._operands[0].llil_read(il),
                    il.const(1, 1),
                    'SVNZC'
                )
            )
        )


class Instruction_BLD(Instruction):
    register_order = ['d', 'b']

    @staticmethod
    def instruction_signature():
        return '1111 100d dddd 0bbb'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.or_expr(
                    1,
                    self._operands[0].llil_read(il),
                    il.shift_left(
                        1,
                        il.flag('T'),
                        self._operands[1].llil_read(il)
                    ),
                ),
            )
        )


class Instruction_BR_Abstract(Instruction):
    @classmethod
    def args_to_operands(cls, chip, args):
        k = args['k']
        if k > 63:
            k -= 128

        return [
            operand.OperandRelativeAddress(chip, k * 2 + 2)
        ]

    @abc.abstractmethod
    def get_llil_condition(self, il):
        pass

    def get_llil(self, il):
        dst = self._operands[0]
        abs_addr = dst.immediate_value + self._addr
        if abs_addr >= self._chip.ROM_SIZE:
            abs_addr -= self._chip.ROM_SIZE
        elif abs_addr < 0:
            abs_addr += self._chip.ROM_SIZE

        f = binaryninja.LowLevelILLabel()
        label = il.get_label_for_address(binaryninja.Architecture['AVR'], abs_addr)
        if label is not None:
            t = label
        else:
            t = binaryninja.LowLevelILLabel()

        il.append(
            il.if_expr(
                self.get_llil_condition(il),
                t,
                f,
            )
        )
        if label is None:
            il.mark_label(t)
            il.append(il.jump(il.const(3, abs_addr)))
        il.mark_label(f)


class Instruction_BRCC(Instruction_BR_Abstract):
    @staticmethod
    def instruction_signature():
        return '1111 01kk kkkk k000'.replace(' ', '')

    def get_llil_condition(self, il):
        return il.flag_condition(LowLevelILFlagCondition.LLFC_UGE)


class Instruction_BRCS(Instruction_BR_Abstract):
    @staticmethod
    def instruction_signature():
        return '1111 00kk kkkk k000'.replace(' ', '')

    def get_llil_condition(self, il):
        return il.flag_condition(LowLevelILFlagCondition.LLFC_ULT)


class Instruction_BREAK(Instruction):
    @staticmethod
    def instruction_signature():
        return '1001 0101 1001 1000'.replace(' ', '')

    def get_llil(self, il):
        il.append(il.breakpoint())


class Instruction_BREQ(Instruction_BR_Abstract):
    @staticmethod
    def instruction_signature():
        return '1111 00kk kkkk k001'.replace(' ', '')

    def get_llil_condition(self, il):
        return il.flag_condition(LowLevelILFlagCondition.LLFC_E)


class Instruction_BRGE(Instruction_BR_Abstract):
    @staticmethod
    def instruction_signature():
        return '1111 01kk kkkk k100'.replace(' ', '')

    def get_llil_condition(self, il):
        return il.flag_condition(LowLevelILFlagCondition.LLFC_SGE)


class Instruction_BRHC(Instruction_BR_Abstract):
    @staticmethod
    def instruction_signature():
        return '1111 01kk kkkk k101'.replace(' ', '')

    def get_llil_condition(self, il):
        return il.compare_equal(1, il.const(1, 0), il.flag('H'))


class Instruction_BRHS(Instruction_BR_Abstract):
    @staticmethod
    def instruction_signature():
        return '1111 00kk kkkk k101'.replace(' ', '')

    def get_llil_condition(self, il):
        return il.compare_equal(1, il.const(1, 1), il.flag('H'))


class Instruction_BRID(Instruction_BR_Abstract):
    @staticmethod
    def instruction_signature():
        return '1111 01kk kkkk k111'.replace(' ', '')

    def get_llil_condition(self, il):
        return il.compare_equal(1, il.const(1, 0), il.flag('I'))


class Instruction_BRIE(Instruction_BR_Abstract):
    @staticmethod
    def instruction_signature():
        return '1111 00kk kkkk k111'.replace(' ', '')

    def get_llil_condition(self, il):
        return il.compare_equal(1, il.const(1, 1), il.flag('I'))


class Instruction_BRLO(Instruction_BR_Abstract):
    @staticmethod
    def instruction_signature():
        return '1111 00kk kkkk k000'.replace(' ', '')

    def get_llil_condition(self, il):
        return il.flag_condition(LowLevelILFlagCondition.LLFC_ULT)


class Instruction_BRLT(Instruction_BR_Abstract):
    @staticmethod
    def instruction_signature():
        return '1111 00kk kkkk k100'.replace(' ', '')

    def get_llil_condition(self, il):
        return il.flag_condition(LowLevelILFlagCondition.LLFC_SLT)

class Instruction_BRMI(Instruction_BR_Abstract):
    @staticmethod
    def instruction_signature():
        return '1111 00kk kkkk k010'.replace(' ', '')

    def get_llil_condition(self, il):
        return il.flag_condition(LowLevelILFlagCondition.LLFC_NEG)


class Instruction_BRNE(Instruction_BR_Abstract):
    @staticmethod
    def instruction_signature():
        return '1111 01kk kkkk k001'.replace(' ', '')

    def get_llil_condition(self, il):
        return il.flag_condition(LowLevelILFlagCondition.LLFC_NE)


class Instruction_BRPL(Instruction_BR_Abstract):
    @staticmethod
    def instruction_signature():
        return '1111 01kk kkkk k010'.replace(' ', '')

    def get_llil_condition(self, il):
        return il.flag_condition(LowLevelILFlagCondition.LLFC_POS)


class Instruction_BRSH(Instruction_BR_Abstract):
    @staticmethod
    def instruction_signature():
        return '1111 01kk kkkk k000'.replace(' ', '')

    def get_llil_condition(self, il):
        return il.flag_condition(LowLevelILFlagCondition.LLFC_UGE)


class Instruction_BRTC(Instruction_BR_Abstract):
    @staticmethod
    def instruction_signature():
        return '1111 01kk kkkk k110'.replace(' ', '')

    def get_llil_condition(self, il):
        return il.compare_equal(1, il.const(1, 0), il.flag('T'))


class Instruction_BRTS(Instruction_BR_Abstract):
    @staticmethod
    def instruction_signature():
        return '1111 00kk kkkk k110'.replace(' ', '')

    def get_llil_condition(self, il):
        return il.compare_equal(1, il.const(1, 1), il.flag('T'))


class Instruction_BRVC(Instruction_BR_Abstract):
    @staticmethod
    def instruction_signature():
        return '1111 01kk kkkk k011'.replace(' ', '')

    def get_llil_condition(self, il):
        return il.flag_condition(LowLevelILFlagCondition.LLFC_NO)


class Instruction_BRVS(Instruction_BR_Abstract):
    @staticmethod
    def instruction_signature():
        return '1111 00kk kkkk k011'.replace(' ', '')

    def get_llil_condition(self, il):
        return il.flag_condition(LowLevelILFlagCondition.LLFC_O)


class Instruction_BST(Instruction):
    register_order = ['d', 'b']

    @staticmethod
    def instruction_signature():
        return '1111 101d dddd 0bbb'.replace(' ', '')

    def get_llil(self, il):
        # TODO: Validate lifting.
        il.append(
            il.set_flag(
                'T',
                il.test_bit(
                    1,
                    self._operands[0].llil_read(il),
                    self._operands[1].llil_read(il)
                )
            )
        )


class Instruction_CALL(Instruction):
    @classmethod
    def args_to_operands(cls, chip, args):
        return [
            operand.OperandDirectAddress(chip, args['k'] * 2)
        ]

    @staticmethod
    def instruction_signature():
        return '1001 010k kkkk 111k kkkk kkkk kkkk kkkk'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.call(self._operands[0].llil_read(il))
        )


class Instruction_CBI(Instruction):
    register_order = ['A', 'b']

    @staticmethod
    def instruction_signature():
        return '1001 1000 AAAA Abbb'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.store(
                1,
                il.const(
                    3,
                    self._chip.get_register_offset(self._operands[0].symbolic_value) + RAM_SEGMENT_BEGIN
                ),
                il.and_expr(
                    1,
                    self._operands[0].llil_read(il),
                    il.not_expr(
                        1,
                        il.shift_left(
                            1,
                            il.const(1, 1),
                            self._operands[1].llil_read(il),
                        )
                    )
                )
            )
        )


class Instruction_CLC(Instruction):
    @staticmethod
    def instruction_signature():
        return '1001 0100 1000 1000'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_flag(
                'C',
                il.const(1, 0)
            )
        )


class Instruction_CLH(Instruction):
    @staticmethod
    def instruction_signature():
        return '1001 0100 1101 1000'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_flag(
                'H',
                il.const(1, 0)
            )
        )


class Instruction_CLI(Instruction):
    @staticmethod
    def instruction_signature():
        return '1001 0100 1111 1000'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_flag(
                'I',
                il.const(1, 0)
            )
        )


class Instruction_CLN(Instruction):
    @staticmethod
    def instruction_signature():
        return '1001 0100 1010 1000'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_flag(
                'N',
                il.const(1, 0)
            )
        )


class Instruction_CLR(Instruction):
    register_order = ['d', 'r']

    def get_instruction_text(self):
        tokens = [
            InstructionTextToken(
                InstructionTextTokenType.InstructionToken, self.name()),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ' '
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                self.operands[0].symbolic_value
            ),
        ]

        return tokens

    @staticmethod
    def verify_args(args):
        return args['d'] == args['r']

    @staticmethod
    def instruction_signature():
        return '0010 01rd dddd rrrr'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.xor_expr(
                    1,
                    self._operands[0].llil_read(il),
                    self._operands[0].llil_read(il)
                )
            )
        )


class Instruction_CLS(Instruction):
    @staticmethod
    def instruction_signature():
        return '1001 0100 1100 1000'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_flag(
                'S',
                il.const(1, 0)
            )
        )


class Instruction_CLT(Instruction):
    @staticmethod
    def instruction_signature():
        return '1001 0100 1110 1000'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_flag(
                'T',
                il.const(1, 0)
            )
        )


class Instruction_CLV(Instruction):
    @staticmethod
    def instruction_signature():
        return '1001 0100 1011 1000'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_flag(
                'V',
                il.const(1, 0)
            )
        )


class Instruction_CLZ(Instruction):
    @staticmethod
    def instruction_signature():
        return '1001 0100 1001 1000'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_flag(
                'Z',
                il.const(1, 0)
            )
        )


class Instruction_COM(Instruction):
    @staticmethod
    def instruction_signature():
        return '1001 010d dddd 0000'.replace(' ', '')

    def get_llil(self, il):
        # One's complement.
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.sub(
                    1,
                    il.const(1, 0xFF),
                    self._operands[0].llil_read(il),
                    flags='SVNZC'
                )
            )
        )


class Instruction_CP(Instruction):
    register_order = ['d', 'r']

    @staticmethod
    def instruction_signature():
        return '0001 01rd dddd rrrr'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.sub(
                1,
                self._operands[0].llil_read(il),
                self._operands[1].llil_read(il),
                flags='HSVNZC'
            )
        )


class Instruction_CPC(Instruction):
    register_order = ['d', 'r']

    @staticmethod
    def instruction_signature():
        return '0000 01rd dddd rrrr'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.sub(
                1,
                il.sub(
                    1,
                    self._operands[0].llil_read(il),
                    self._operands[1].llil_read(il),
                ),
                il.flag('C'),
                flags='HSVNZC'
            )
        )


class Instruction_CPI(Instruction):
    @classmethod
    def args_to_operands(cls, chip, args):
        k = args['K']
        d = args['d']

        return [
            operand.OperandRegister(chip, d + 16),
            operand.OperandImmediate(chip, k)
        ]

    @staticmethod
    def instruction_signature():
        return '0011 KKKK dddd KKKK'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.sub(
                1,
                self._operands[0].llil_read(il),
                self._operands[1].llil_read(il),
                flags='HSVNZC'
            )
        )


class Instruction_CPSE(Instruction):
    # The IL instruction is non-trivial as it skips one instruction. How are we
    # supposed to know how big the instruction is?
    @staticmethod
    def instruction_signature():
        return '0001 00rd dddd rrrr'.replace(' ', '')

    def get_llil(self, il):
        if len(self._data) == 2:
            # We only got this instruction, cannot determine length of next ins.
            # Assume next instruction has two bytes
            next_len = 2
            binaryninja.log.log_warn(
                "0x{:X}: Lifting: CPSE: We only got 2 bytes but we need more to predict the length of the next instruction".format(self._addr))
        else:
            next_ins = parse_instruction(self._chip, self._addr, self._data[2:])
            if next_ins:
                next_len = next_ins.length()
            else:
                next_len = 2
                binaryninja.log.log_warn(
                    "0x{:X}: Lifting: CPSE: Next instruction invalid?".format(self._addr))

        f = binaryninja.LowLevelILLabel()
        label = il.get_label_for_address(binaryninja.Architecture['AVR'], self._addr + 2 + next_len)
        if label is not None:
            t = label
        else:
            t = binaryninja.LowLevelILLabel()
        il.append(
            il.if_expr(
                il.compare_equal(
                    1,
                    self._operands[0].llil_read(il),
                    self._operands[1].llil_read(il)
                ),
                t,
                f,
            )
        )

        if label is None:
            il.mark_label(t)
            il.append(il.jump(il.const(3, self._addr + 2 + next_len)))
        il.mark_label(f)


class Instruction_DEC(Instruction):
    @staticmethod
    def instruction_signature():
        return '1001 010d dddd 1010'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.sub(
                    1,
                    self._operands[0].llil_read(il),
                    il.const(1, 1),
                    flags='SVNZ'
                )
            )
        )


class Instruction_EICALL(Instruction):
    @staticmethod
    def instruction_signature():
        return '1001 0101 0001 1001'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.call(
                il.or_expr(
                    3,
                    il.shift_left(
                        3,
                        il.load(
                            1,
                            il.const(
                                3,
                                self._chip.get_register_offset(
                                    'EIND') + RAM_SEGMENT_BEGIN
                            )
                        ),
                        il.const(1, 16)
                    ),
                    get_xyz_register(il, self._chip, 'Z', do_ramp=False)
                )
            )
        )


class Instruction_EIJMP(Instruction):
    @staticmethod
    def instruction_signature():
        return '1001 0100 0001 1001'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.jump(
                il.or_expr(
                    3,
                    il.shift_left(
                        3,
                        il.load(
                            1,
                            il.const(
                                3,
                                self._chip.get_register_offset(
                                    'EIND') + RAM_SEGMENT_BEGIN
                            )
                        ),
                        il.const(1, 16)
                    ),
                    get_xyz_register(il, self._chip, 'Z', do_ramp=False)
                )
            )
        )


class Instruction_ELPM_I(Instruction):
    @classmethod
    def name(cls):
        return 'elpm'

    @staticmethod
    def instruction_signature():
        return '1001 0101 1101 1000'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                'r0',
                il.load(
                    1,
                    get_xyz_register(il, self._chip, 'Z')
                )
            )
        )


class Instruction_ELPM_II(Instruction):
    @classmethod
    def name(cls):
        return 'elpm'

    @staticmethod
    def instruction_signature():
        return '1001 000d dddd 0110'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.load(
                    1,
                    get_xyz_register(il, self._chip, 'Z')
                )
            )
        )


class Instruction_ELPM_III(Instruction):
    @classmethod
    def name(cls):
        return 'elpm+'

    @staticmethod
    def instruction_signature():
        return '1001 000d dddd 0111'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.load(
                    1,
                    get_xyz_register(il, self._chip, 'Z')
                )
            )
        )

        il.append(
            il.set_reg(
                2,
                'Z',
                il.add(
                    2,
                    il.reg(2, 'Z'),
                    il.const(2, 1)
                )
            )
        )


class Instruction_EOR(Instruction):
    register_order = ['d', 'r']

    @staticmethod
    def instruction_signature():
        return '0010 01rd dddd rrrr'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.xor_expr(
                    1,
                    self._operands[0].llil_read(il),
                    self._operands[1].llil_read(il),
                    flags='SVNZ'
                )
            )
        )


class Instruction_FMUL(Instruction):
    @classmethod
    def args_to_operands(cls, chip, args):
        r = args['r']
        d = args['d']

        return [
            operand.OperandRegister(chip, d + 16),
            operand.OperandRegister(chip, r + 16)
        ]

    @staticmethod
    def instruction_signature():
        return '0000 0011 0ddd 1rrr'.replace(' ', '')

    # TODO: get_llil(), flags='ZC'


class Instruction_FMULS(Instruction):
    @classmethod
    def args_to_operands(cls, chip, args):
        r = args['r']
        d = args['d']

        return [
            operand.OperandRegister(chip, d + 16),
            operand.OperandRegister(chip, r + 16)
        ]

    @staticmethod
    def instruction_signature():
        return '0000 0011 1ddd 0rrr'.replace(' ', '')

    # TODO: get_llil(), flags='ZC'


class Instruction_FMULSU(Instruction):
    @classmethod
    def args_to_operands(cls, chip, args):
        r = args['r']
        d = args['d']

        return [
            operand.OperandRegister(chip, d + 16),
            operand.OperandRegister(chip, r + 16)
        ]

    @staticmethod
    def instruction_signature():
        return '0000 0011 1ddd 1rrr'.replace(' ', '')

    # TODO: get_llil(), flags='ZC'??? VFY


class Instruction_ICALL(Instruction):
    # Calls Z

    @staticmethod
    def instruction_signature():
        return '1001 0101 0000 1001'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.call(
                get_xyz_register(il, self._chip, 'Z')
            )
        )


class Instruction_IJMP(Instruction):
    # Jumps to Z

    @staticmethod
    def instruction_signature():
        return '1001 0100 0000 1001'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.jump(
                get_xyz_register(il, self._chip, 'Z')
            )
        )


class Instruction_IN(Instruction):
    register_order = ['d', 'A']

    @staticmethod
    def instruction_signature():
        return '1011 0AAd dddd AAAA'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                self._operands[1].llil_read(il)
            )
        )


class Instruction_INC(Instruction):
    @staticmethod
    def instruction_signature():
        return '1001 010d dddd 0011'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.add(
                    1,
                    self._operands[0].llil_read(il),
                    il.const(1, 1),
                    flags='SVNZ'
                )
            )
        )


class Instruction_JMP(Instruction):
    @classmethod
    def args_to_operands(cls, chip, args):
        return [
            operand.OperandDirectAddress(chip, args['k'] * 2)
        ]

    @staticmethod
    def instruction_signature():
        return '1001 010k kkkk 110k kkkk kkkk kkkk kkkk'.replace(' ', '')

    def get_llil(self, il):
        addr = self._operands[0].immediate_value
        label = il.get_label_for_address(binaryninja.Architecture['AVR'], addr)
        if label:
            il.append(il.goto(label))
        else:
            il.append(il.jump(self._operands[0].llil_read(il)))


class Instruction_LAC(Instruction):
    @staticmethod
    def instruction_signature():
        return '1001 001r rrrr 0110'.replace(' ', '')


class Instruction_LAS(Instruction):
    @staticmethod
    def instruction_signature():
        return '1001 001r rrrr 0101'.replace(' ', '')


class Instruction_LAT(Instruction):
    @staticmethod
    def instruction_signature():
        return '1001 001r rrrr 0111'.replace(' ', '')


class Instruction_LD_X_I(Instruction):
    @classmethod
    def name(cls):
        return "ld"

    def get_instruction_text(self):
        tokens = [
            InstructionTextToken(
                InstructionTextTokenType.InstructionToken,
                self.name()
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ' '
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                self.operands[0].symbolic_value
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ', '
            ),
            InstructionTextToken(
                InstructionTextTokenType.BeginMemoryOperandToken,
                '['
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                'X'
            ),
            InstructionTextToken(
                InstructionTextTokenType.EndMemoryOperandToken,
                ']'
            ),
        ]

        return tokens

    @staticmethod
    def instruction_signature():
        return '1001 000d dddd 1100'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.load(
                    1,
                    il.add(
                        3,
                        il.const_pointer(3, RAM_SEGMENT_BEGIN),
                        get_xyz_register(il, self._chip, 'X')
                    )
                )
            )
        )


class Instruction_LD_X_II(Instruction):
    @classmethod
    def name(cls):
        return "ld"

    def get_instruction_text(self):
        tokens = [
            InstructionTextToken(
                InstructionTextTokenType.InstructionToken, self.name()),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ' '
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                self.operands[0].symbolic_value
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ', '
            ),
            InstructionTextToken(
                InstructionTextTokenType.BeginMemoryOperandToken,
                '['
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                'X'
            ),
            InstructionTextToken(
                InstructionTextTokenType.StringToken,
                '+'
            ),
            InstructionTextToken(
                InstructionTextTokenType.EndMemoryOperandToken,
                ']'
            ),
        ]

        return tokens

    @staticmethod
    def instruction_signature():
        return '1001 000d dddd 1101'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.load(
                    1,
                    il.add(
                        3,
                        il.const_pointer(3, RAM_SEGMENT_BEGIN),
                        get_xyz_register(il, self._chip, 'X')
                    )
                )
            )
        )
        # X++
        il.append(
            il.set_reg(
                2,
                'X',
                il.add(
                    2,
                    il.reg(2, 'X'),
                    il.const(2, 1)
                )
            )
        )


class Instruction_LD_X_III(Instruction):
    @classmethod
    def name(cls):
        return "ld"

    def get_instruction_text(self):
        tokens = [
            InstructionTextToken(
                InstructionTextTokenType.InstructionToken, self.name()),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ' '
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                self.operands[0].symbolic_value
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ', '
            ),
            InstructionTextToken(
                InstructionTextTokenType.BeginMemoryOperandToken,
                '['
            ),
            InstructionTextToken(
                InstructionTextTokenType.StringToken,
                '-'
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                'X'
            ),
            InstructionTextToken(
                InstructionTextTokenType.EndMemoryOperandToken,
                ']'
            ),
        ]

        return tokens

    @staticmethod
    def instruction_signature():
        return '1001 000d dddd 1110'.replace(' ', '')

    def get_llil(self, il):
        # --X
        il.append(
            il.set_reg(
                2,
                'X',
                il.sub(
                    2,
                    il.reg(2, 'X'),
                    il.const(2, 1)
                )
            )
        )

        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.load(
                    1,
                    il.add(
                        3,
                        il.const_pointer(3, RAM_SEGMENT_BEGIN),
                        get_xyz_register(il, self._chip, 'X')
                    )
                )
            )
        )


class Instruction_LD_Y_I(Instruction):
    @classmethod
    def name(cls):
        return "ld"

    def get_instruction_text(self):
        tokens = [
            InstructionTextToken(
                InstructionTextTokenType.InstructionToken, self.name()),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ' '
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                self.operands[0].symbolic_value
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ', '
            ),
            InstructionTextToken(
                InstructionTextTokenType.BeginMemoryOperandToken,
                '['
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                'Y'
            ),
            InstructionTextToken(
                InstructionTextTokenType.EndMemoryOperandToken,
                ']'
            ),
        ]

        return tokens

    @staticmethod
    def instruction_signature():
        return '1000 000d dddd 1000'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.load(
                    1,
                    il.add(
                        3,
                        il.const_pointer(3, RAM_SEGMENT_BEGIN),
                        get_xyz_register(il, self._chip, 'Y')
                    )
                )
            )
        )


class Instruction_LD_Y_II(Instruction):
    @classmethod
    def name(cls):
        return "ld"

    def get_instruction_text(self):
        tokens = [
            InstructionTextToken(
                InstructionTextTokenType.InstructionToken, self.name()),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ' '
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                self.operands[0].symbolic_value
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ', '
            ),
            InstructionTextToken(
                InstructionTextTokenType.BeginMemoryOperandToken,
                '['
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                'Y'
            ),
            InstructionTextToken(
                InstructionTextTokenType.StringToken,
                '+'
            ),
            InstructionTextToken(
                InstructionTextTokenType.EndMemoryOperandToken,
                ']'
            ),
        ]

        return tokens

    @staticmethod
    def instruction_signature():
        return '1001 000d dddd 1001'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.load(
                    1,
                    il.add(
                        3,
                        il.const_pointer(3, RAM_SEGMENT_BEGIN),
                        get_xyz_register(il, self._chip, 'Y')
                    )
                )
            )
        )

        # Y++?
        il.append(
            il.set_reg(
                2,
                'Y',
                il.add(
                    2,
                    il.reg(2, 'Y'),
                    il.const(2, 1)
                )
            )
        )


class Instruction_LD_Y_III(Instruction):
    @classmethod
    def name(cls):
        return "ld"

    def get_instruction_text(self):
        tokens = [
            InstructionTextToken(
                InstructionTextTokenType.InstructionToken, self.name()),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ' '
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                self.operands[0].symbolic_value
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ', '
            ),
            InstructionTextToken(
                InstructionTextTokenType.BeginMemoryOperandToken,
                '['
            ),
            InstructionTextToken(
                InstructionTextTokenType.StringToken,
                '-'
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                'Y'
            ),
            InstructionTextToken(
                InstructionTextTokenType.EndMemoryOperandToken,
                ']'
            ),
        ]

        return tokens

    @staticmethod
    def instruction_signature():
        return '1001 000d dddd 1010'.replace(' ', '')

    def get_llil(self, il):
        # --Y
        il.append(
            il.set_reg(
                2,
                'Y',
                il.sub(
                    2,
                    il.reg(2, 'Y'),
                    il.const(2, 1)
                )
            )
        )
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.load(
                    1,
                    il.add(
                        3,
                        il.const_pointer(3, RAM_SEGMENT_BEGIN),
                        get_xyz_register(il, self._chip, 'Y')
                    )
                )
            )
        )


class Instruction_LD_Y_IV(Instruction):
    register_order = ['d', 'q']

    @classmethod
    def name(cls):
        return "ld"

    def get_instruction_text(self):
        tokens = [
            InstructionTextToken(
                InstructionTextTokenType.InstructionToken, self.name()),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ' '
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                self.operands[0].symbolic_value
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ', '
            ),
            InstructionTextToken(
                InstructionTextTokenType.BeginMemoryOperandToken,
                '['
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                'Y'
            ),
            InstructionTextToken(
                InstructionTextTokenType.StringToken,
                ' + '
            ),
            InstructionTextToken(
                InstructionTextTokenType.IntegerToken,
                str(self.operands[1].immediate_value)
            ),
            InstructionTextToken(
                InstructionTextTokenType.EndMemoryOperandToken,
                ']'
            ),
        ]

        return tokens

    @staticmethod
    def instruction_signature():
        return '10q0 qq0d dddd 1qqq'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.load(
                    1,
                    il.add(
                        3,
                        il.const_pointer(3, RAM_SEGMENT_BEGIN),
                        il.zero_extend(3, il.add(
                            3,
                            self._operands[1].llil_read(il),
                            get_xyz_register(il, self._chip, 'Y')
                        ))
                    ),
                )
            )
        )


class Instruction_LD_Z_I(Instruction):
    @classmethod
    def name(cls):
        return "ld"

    def get_instruction_text(self):
        tokens = [
            InstructionTextToken(
                InstructionTextTokenType.InstructionToken,
                self.name()
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ' '
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                self.operands[0].symbolic_value
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ', '
            ),
            InstructionTextToken(
                InstructionTextTokenType.BeginMemoryOperandToken,
                '['
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                'Z'
            ),
            InstructionTextToken(
                InstructionTextTokenType.EndMemoryOperandToken,
                ']'
            ),
        ]

        return tokens

    @staticmethod
    def instruction_signature():
        return '1000 000d dddd 0000'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.load(
                    1,
                    il.add(
                        3,
                        il.const_pointer(3, RAM_SEGMENT_BEGIN),
                        get_xyz_register(il, self._chip, 'Z')
                    )
                )
            )
        )


class Instruction_LD_Z_II(Instruction):
    @classmethod
    def name(cls):
        return "ld"

    def get_instruction_text(self):
        tokens = [
            InstructionTextToken(
                InstructionTextTokenType.InstructionToken, self.name()),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ' '
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                self.operands[0].symbolic_value
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ', '
            ),
            InstructionTextToken(
                InstructionTextTokenType.BeginMemoryOperandToken,
                '['
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                'Z'
            ),
            InstructionTextToken(
                InstructionTextTokenType.StringToken,
                '+'
            ),
            InstructionTextToken(
                InstructionTextTokenType.EndMemoryOperandToken,
                ']'
            ),
        ]

        return tokens

    @staticmethod
    def instruction_signature():
        return '1001 000d dddd 0001'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.load(
                    1,
                    il.add(
                        3,
                        il.const_pointer(3, RAM_SEGMENT_BEGIN),
                        get_xyz_register(il, self._chip, 'Z')
                    )
                )
            )
        )

        il.append(
            il.set_reg(
                2,
                'Z',
                il.add(
                    2,
                    il.reg(2, 'Z'),
                    il.const(2, 1)
                )
            )
        )


class Instruction_LD_Z_III(Instruction):
    @classmethod
    def name(cls):
        return "ld"

    def get_instruction_text(self):
        tokens = [
            InstructionTextToken(
                InstructionTextTokenType.InstructionToken, self.name()),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ' '
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                self.operands[0].symbolic_value
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ', '
            ),
            InstructionTextToken(
                InstructionTextTokenType.BeginMemoryOperandToken,
                '['
            ),
            InstructionTextToken(
                InstructionTextTokenType.StringToken,
                '-'
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                'Z'
            ),
            InstructionTextToken(
                InstructionTextTokenType.EndMemoryOperandToken,
                ']'
            ),
        ]

        return tokens

    @staticmethod
    def instruction_signature():
        return '1001 000d dddd 0010'.replace(' ', '')

    def get_llil(self, il):
        # --Z
        il.append(
            il.set_reg(
                2,
                'Z',
                il.sub(
                    2,
                    il.reg(2, 'Z'),
                    il.const(2, 1)
                )
            )
        )

        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.load(
                    1,
                    il.add(
                        3,
                        il.const_pointer(3, RAM_SEGMENT_BEGIN),
                        get_xyz_register(il, self._chip, 'Z')
                    )
                )
            )
        )


class Instruction_LD_Z_IV(Instruction):
    register_order = ['d', 'q']

    @classmethod
    def name(cls):
        return "ld"

    def get_instruction_text(self):
        tokens = [
            InstructionTextToken(
                InstructionTextTokenType.InstructionToken, self.name()),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ' '
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                self.operands[0].symbolic_value
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ', '
            ),
            InstructionTextToken(
                InstructionTextTokenType.BeginMemoryOperandToken,
                '['
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                'Z'
            ),
            InstructionTextToken(
                InstructionTextTokenType.StringToken,
                ' + '
            ),
            InstructionTextToken(
                InstructionTextTokenType.IntegerToken,
                str(self.operands[1].immediate_value)
            ),
            InstructionTextToken(
                InstructionTextTokenType.EndMemoryOperandToken,
                ']'
            ),
        ]

        return tokens

    @staticmethod
    def instruction_signature():
        return '10q0 qq0d dddd 0qqq'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.load(
                    1,
                    il.add(
                        3,
                        il.const_pointer(3, RAM_SEGMENT_BEGIN),
                        il.zero_extend(3, il.add(
                            3,
                            self._operands[1].llil_read(il),
                            get_xyz_register(il, self._chip, 'Z')
                        ))
                    ),
                )
            )
        )


class Instruction_LDI(Instruction):
    @classmethod
    def args_to_operands(cls, chip, args):
        d = args['d']
        K = args['K']

        return [
            operand.OperandRegister(chip, d + 16),
            operand.OperandImmediate(chip, K)
        ]

    @staticmethod
    def instruction_signature():
        return '1110 KKKK dddd KKKK'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                self._operands[1].llil_read(il)
            )
        )


class Instruction_LDS_32(Instruction):
    register_order = ['d', 'k']

    @classmethod
    def name(cls):
        return "lds"

    @staticmethod
    def instruction_signature():
        return '1001 000d dddd 0000 kkkk kkkk kkkk kkkk'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.load(
                    1,
                    il.add(
                        3,
                        il.const_pointer(3, RAM_SEGMENT_BEGIN),
                        il.zero_extend(3, self._operands[1].llil_read(il))
                    )
                )
            )
        )

    def get_instruction_text(self):
        tokens = [
            InstructionTextToken(
                InstructionTextTokenType.InstructionToken, self.name()),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ' '
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                self.operands[0].symbolic_value
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ', '
            ),
            InstructionTextToken(
                InstructionTextTokenType.PossibleAddressToken,
                hex(self.operands[1].immediate_value + RAM_SEGMENT_BEGIN),
                value=self.operands[1].immediate_value + RAM_SEGMENT_BEGIN
            )
        ]

        return tokens


class Instruction_LDS_16(Instruction):
    @classmethod
    def args_to_operands(cls, chip, args):
        d = args['d']
        k = args['k']

        return [
            operand.OperandRegister(chip, d + 16),
            operand.OperandImmediate(chip, k + 0x40)
        ]

    def name(cls):
        return "lds"

    @staticmethod
    def instruction_signature():
        return '1010 0kkk dddd kkkk'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.load(
                    1,
                    il.add(
                        3,
                        il.const_pointer(3, RAM_SEGMENT_BEGIN),
                        il.zero_extend(3, self._operands[1].llil_read(il))
                    )
                )
            )
        )

    def get_instruction_text(self):
        tokens = [
            InstructionTextToken(
                InstructionTextTokenType.InstructionToken, self.name()),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ' '
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                self.operands[0].symbolic_value
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ', '
            ),
            InstructionTextToken(
                InstructionTextTokenType.PossibleAddressToken,
                hex(self.operands[1].immediate_value + RAM_SEGMENT_BEGIN),
                value=self.operands[1].immediate_value + RAM_SEGMENT_BEGIN
            )
        ]

        return tokens


class Instruction_LPM_I(Instruction):
    @classmethod
    def name(cls):
        return "lpm"

    @staticmethod
    def instruction_signature():
        return '1001 0101 1100 1000'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                'r0',
                il.load(
                    1,
                    get_xyz_register(il, self._chip, 'Z')
                )
            )
        )


class Instruction_LPM_II(Instruction):
    @classmethod
    def name(cls):
        return "lpm"

    def get_instruction_text(self):
        tokens = [
            InstructionTextToken(
                InstructionTextTokenType.InstructionToken,
                self.name()
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ' '
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                self.operands[0].symbolic_value
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ', '
            ),
            InstructionTextToken(
                InstructionTextTokenType.BeginMemoryOperandToken,
                '['
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                'Z'
            ),
            InstructionTextToken(
                InstructionTextTokenType.EndMemoryOperandToken,
                ']'
            ),
        ]

        return tokens

    @staticmethod
    def instruction_signature():
        return '1001 000d dddd 0100'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.load(
                    1,
                    get_xyz_register(il, self._chip, 'Z')
                )
            )
        )


class Instruction_LPM_III(Instruction):
    @classmethod
    def name(cls):
        return "lpm"

    def get_instruction_text(self):
        tokens = [
            InstructionTextToken(
                InstructionTextTokenType.InstructionToken,
                self.name()
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ' '
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                self.operands[0].symbolic_value
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ', '
            ),
            InstructionTextToken(
                InstructionTextTokenType.BeginMemoryOperandToken,
                '['
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                'Z'
            ),
            InstructionTextToken(
                InstructionTextTokenType.StringToken,
                '+'
            ),
            InstructionTextToken(
                InstructionTextTokenType.EndMemoryOperandToken,
                ']'
            ),
        ]

        return tokens
    @staticmethod
    def instruction_signature():
        return '1001 000d dddd 0101'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.load(
                    1,
                    get_xyz_register(il, self._chip, 'Z')
                )
            )
        )
        il.append(
            il.set_reg(
                2,
                'Z',
                il.add(
                    2,
                    il.reg(2, 'Z'),
                    il.const(2, 1)
                )
            )
        )


class Instruction_LSL(Instruction):
    register_order = ['d', 'r']

    def get_instruction_text(self):
        tokens = [
            InstructionTextToken(
                InstructionTextTokenType.InstructionToken, self.name()),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ' '
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                self.operands[0].symbolic_value
            ),
        ]

        return tokens

    @staticmethod
    def verify_args(args):
        return args['d'] == args['r']

    @staticmethod
    def instruction_signature():
        return '0000 11rd dddd rrrr'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.shift_left(
                    1,
                    self._operands[0].llil_read(il),
                    il.const(1, 1),
                    flags='HSVNZC'
                )
            )
        )


class Instruction_LSR(Instruction):
    @staticmethod
    def instruction_signature():
        return '1001 010d dddd 0110'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.logical_shift_right(
                    1,
                    self._operands[0].llil_read(il),
                    il.const(1, 1),
                    flags='SVNZC'
                )
            )
        )


class Instruction_MOV(Instruction):
    register_order = ['d', 'r']

    @staticmethod
    def instruction_signature():
        return '0010 11rd dddd rrrr'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                self._operands[1].llil_read(il)
            )
        )


class Instruction_MOVW(Instruction):
    @classmethod
    def args_to_operands(cls, chip, args):
        d = args['d']
        r = args['r']

        return [
            operand.OperandRegisterWide(chip, d * 2),
            operand.OperandRegisterWide(chip, r * 2)
        ]

    @staticmethod
    def instruction_signature():
        return '0000 0001 dddd rrrr'.replace(' ', '')

    def get_llil(self, il):
        dlow = self._operands[0].low()
        dhigh = self._operands[0].high()
        rlow = self._operands[1].low()
        rhigh = self._operands[1].high()
        il.append(il.set_reg(1, dlow, il.reg(1, rlow)))
        il.append(il.set_reg(1, dhigh, il.reg(1, rhigh)))


class Instruction_MUL(Instruction):
    register_order = ['d', 'r']

    @staticmethod
    def instruction_signature():
        return '1001 11rd dddd rrrr'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg_split(
                1,
                'r1',
                'r0',
                il.mult(
                    2,
                    il.zero_extend(2, self._operands[0].llil_read(il)),
                    il.zero_extend(2, self._operands[1].llil_read(il)),
                    flags='ZC'
                ),
            )
        )


class Instruction_MULS(Instruction):
    @classmethod
    def args_to_operands(cls, chip, args):
        d = args['d']
        r = args['r']

        return [
            operand.OperandRegister(chip, d + 16),
            operand.OperandRegister(chip, r + 16)
        ]

    @staticmethod
    def instruction_signature():
        return '0000 0010 dddd rrrr'.replace(' ', '')

    def get_llil(self, il):
        # TODO: verify signed result.
        il.append(
            il.set_reg_split(
                1,
                'r1',
                'r0',
                il.mult(
                    2,
                    il.zero_extend(2, self._operands[0].llil_read(il)),
                    il.zero_extend(2, self._operands[1].llil_read(il)),
                    flags='ZC'
                ),
            )
        )


class Instruction_MULSU(Instruction):
    @classmethod
    def args_to_operands(cls, chip, args):
        d = args['d']
        r = args['r']

        return [
            operand.OperandRegister(chip, d + 16),
            operand.OperandRegister(chip, r + 16)
        ]

    @staticmethod
    def instruction_signature():
        return '0000 0011 0ddd 0rrr'.replace(' ', '')

    def get_llil(self, il):
        # TODO: verify signed/unsigned result
        il.append(
            il.set_reg_split(
                1,
                'r1',
                'r0',
                il.mult(
                    2,
                    il.zero_extend(2, self._operands[0].llil_read(il)),
                    il.zero_extend(2, self._operands[1].llil_read(il)),
                    flags='ZC'
                ),
            )
        )


class Instruction_NEG(Instruction):
    @staticmethod
    def instruction_signature():
        return '1001 010d dddd 0001'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.neg_expr(
                    1,
                    self._operands[0].llil_read(il)
                ),
                flags='HSVNZC'
            )
        )


class Instruction_NOP(Instruction):
    @staticmethod
    def instruction_signature():
        return '0000 0000 0000 0000'.replace(' ', '')

    def get_llil(self, il):
        il.append(il.nop())


class Instruction_OR(Instruction):
    register_order = ['d', 'r']

    @staticmethod
    def instruction_signature():
        return '0010 10rd dddd rrrr'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.or_expr(
                    1,
                    self._operands[0].llil_read(il),
                    self._operands[1].llil_read(il)
                ),
                flags='HSVNZ'
            )
        )


class Instruction_ORI(Instruction):
    @classmethod
    def args_to_operands(cls, chip, args):
        d = args['d']
        K = args['K']

        return [
            operand.OperandRegister(chip, d + 16),
            operand.OperandImmediate(chip, K)
        ]

    @staticmethod
    def instruction_signature():
        return '0110 KKKK dddd KKKK'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.or_expr(
                    1,
                    self._operands[0].llil_read(il),
                    self._operands[1].llil_read(il)
                ),
                flags='HSVNZ'
            )
        )


class Instruction_OUT(Instruction):
    register_order = ['A', 'r']

    @staticmethod
    def instruction_signature():
        return '1011 1AAr rrrr AAAA'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.store(
                1,
                il.const(
                    3,
                    self._chip.get_register_offset(self._operands[0].symbolic_value) + RAM_SEGMENT_BEGIN
                ),
                self._operands[1].llil_read(il)
            )
        )


class Instruction_POP(Instruction):
    @staticmethod
    def instruction_signature():
        return '1001 000d dddd 1111'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.pop(1)
            )
        )


class Instruction_PUSH(Instruction):
    @staticmethod
    def instruction_signature():
        return '1001 001d dddd 1111'.replace(' ', '')

    def get_llil(self, il):
        il.append(il.push(1, self._operands[0].llil_read(il)))


class Instruction_RCALL(Instruction):
    @classmethod
    def args_to_operands(cls, chip, args):
        k = args['k']
        if k > 2047:
            k -= 4096
        return [
            operand.OperandRelativeAddress(chip, k * 2 + 2)
        ]

    @staticmethod
    def instruction_signature():
        return '1101 kkkk kkkk kkkk'.replace(' ', '')

    def get_llil(self, il):
        v = self._operands[0].immediate_value + self._addr
        if v == (self._addr + 2):
            il.append(il.push(2, il.const(2, v)))
            return
        if v >= self._chip.ROM_SIZE:
            v -= self._chip.ROM_SIZE
        elif v < 0:
            v += self._chip.ROM_SIZE

        il.append(il.call(il.const(3, v)))


class Instruction_RET(Instruction):
    @staticmethod
    def instruction_signature():
        return '1001 0101 0000 1000'.replace(' ', '')

    def get_llil(self, il):
        il.append(il.ret(il.pop(2)))


class Instruction_RETI(Instruction):
    @staticmethod
    def instruction_signature():
        return '1001 0101 0001 1000'.replace(' ', '')

    def get_llil(self, il):
        il.append(il.set_flag('I', il.const(1, 0)))
        il.append(il.ret(il.pop(2)))


class Instruction_RJMP(Instruction):
    @classmethod
    def args_to_operands(cls, chip, args):
        k = args['k']
        if k >= 2 * 1024:
            k -= 4 * 1024

        return [
            operand.OperandRelativeAddress(chip, k * 2 + 2)
        ]

    @staticmethod
    def instruction_signature():
        return '1100 kkkk kkkk kkkk'.replace(' ', '')

    def get_llil(self, il):
        taddr = self._operands[0].immediate_value + self._addr
        if taddr < 0:
            taddr += self._chip.ROM_SIZE
        if taddr >= self._chip.ROM_SIZE:
            taddr -= self._chip.ROM_SIZE

        label = il.get_label_for_address(binaryninja.Architecture['AVR'], taddr)
        if label:
            il.append(il.goto(label))
        else:
            il.append(il.jump(il.const(3, taddr)))


class Instruction_ROL(Instruction):
    register_order = ['d', 'r']

    def get_instruction_text(self):
        tokens = [
            InstructionTextToken(
                InstructionTextTokenType.InstructionToken, self.name()),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ' '
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                self.operands[0].symbolic_value
            ),
        ]

        return tokens

    @staticmethod
    def verify_args(args):
        return args['d'] == args['r']

    @staticmethod
    def instruction_signature():
        return '0001 11rd dddd rrrr'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.rotate_left_carry(
                    1,
                    self._operands[0].llil_read(il),
                    il.const(1, 1),
                    il.flag('C'),
                    flags='HSVNZC',
                )
            )
        )


class Instruction_ROR(Instruction):
    @staticmethod
    def instruction_signature():
        return '1001 010d dddd 0111'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.rotate_right_carry(
                    1,
                    self._operands[0].llil_read(il),
                    il.const(1, 1),
                    il.flag('C'),
                    flags='HSVNZC',
                )
            )
        )


class Instruction_SBC(Instruction):
    register_order = ['d', 'r']

    @staticmethod
    def instruction_signature():
        return '0000 10rd dddd rrrr'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.sub(
                    1,
                    self._operands[0].llil_read(il),
                    il.add(
                        1,
                        self._operands[1].llil_read(il),
                        il.flag('C')
                    ),
                    flags='HSVNZC'
                )
            )
        )


class Instruction_SBCI(Instruction):
    @classmethod
    def args_to_operands(cls, chip, args):
        k = args['K']
        d = args['d']

        return [
            operand.OperandRegister(chip, d + 16),
            operand.OperandImmediate(chip, k)
        ]

    @staticmethod
    def instruction_signature():
        return '0100 KKKK dddd KKKK'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.sub(
                    1,
                    self._operands[0].llil_read(il),
                    self._operands[1].llil_read(il),
                    flags='HSVNZC'
                )
            )
        )


class Instruction_SBI(Instruction):
    register_order = ['A', 'b']

    @staticmethod
    def instruction_signature():
        return '1001 1010 AAAA Abbb'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.store(
                1,
                il.const(
                    3,
                    self._chip.get_register_offset(self._operands[0].symbolic_value) + RAM_SEGMENT_BEGIN
                ),
                il.or_expr(
                    1,
                    self._operands[0].llil_read(il),
                    il.shift_left(
                        1,
                        il.const(1, 1),
                        self._operands[1].llil_read(il),
                    )
                )
            )
        )


class Instruction_SkipInstruction_Abstract(Instruction):
    """
    We're currently assuming that the instruction-to-be-skipped is one
    instruction big. otherwise we're screwed. I don't know how we could solve
    this at this moment. TODO.
    """
    @abc.abstractmethod
    def _get_llil_condition(self, il):
        pass

    def get_llil(self, il):
        if len(self._data) == 2:
            # We only got this instruction, cannot determine length of next ins.
            # Assume next instruction has two bytes
            next_len = 2
            binaryninja.log.log_warn(
                "0x{:X}: Lifting: SB**: We only got 2 bytes but we need more to predict the length of the next instruction".format(self._addr))
        else:
            next_ins = parse_instruction(self._chip, self._addr, self._data[2:])
            if next_ins:
                next_len = next_ins.length()
            else:
                next_len = 2
                binaryninja.log.log_warn(
                    "0x{:X}: Lifting: SB**: Next instruction invalid?".format(self._addr))

        f = binaryninja.LowLevelILLabel()
        label = il.get_label_for_address(binaryninja.Architecture['AVR'], self._addr + 2 + next_len)
        if label is not None:
            t = label
        else:
            t = binaryninja.LowLevelILLabel()

        il.append(
            il.if_expr(
                self._get_llil_condition(il),
                t,
                f,
            )
        )

        if label is None:
            il.mark_label(t)
            il.append(il.jump(il.const(3, self._addr + 2 + next_len)))
        il.mark_label(f)


class Instruction_SBIC(Instruction_SkipInstruction_Abstract):
    @staticmethod
    def instruction_signature():
        return '1001 1001 AAAA Abbb'.replace(' ', '')

    def _get_llil_condition(self, il):
        return il.compare_equal(
            1,
            il.test_bit(
                1,
                self._operands[0].llil_read(il),
                self._operands[1].llil_read(il)
            ),
            il.const(1, 0)
        )


class Instruction_SBIS(Instruction_SkipInstruction_Abstract):
    @staticmethod
    def instruction_signature():
        return '1001 1011 AAAA Abbb'.replace(' ', '')

    def _get_llil_condition(self, il):
        return il.compare_equal(
            1,
            il.test_bit(
                1,
                self._operands[0].llil_read(il),
                self._operands[1].llil_read(il)
            ),
            il.const(1, 1)
        )


class Instruction_SBIW(Instruction):
    @classmethod
    def args_to_operands(cls, chip, args):
        d = args['d']
        K = args['K']

        return [
            operand.OperandRegisterWide(chip, d * 2 + 24),
            operand.OperandImmediate(chip, K)
        ]

    @staticmethod
    def instruction_signature():
        return '1001 0111 KKdd KKKK'.replace(' ', '')

    def get_llil(self, il):
        rhigh, rlow = self._operands[0].symbolic_value.split(':')
        il.append(
            il.set_reg_split(
                1,
                rhigh, rlow,
                il.sub(
                    2,
                    get_reg_pair(il, rhigh, rlow),
                    self._operands[1].llil_read(il),
                    'SVNZC'
                )
            )
        )


class Instruction_SBR(Instruction):
    @classmethod
    def args_to_operands(cls, chip, args):
        d = args['d']
        K = args['K']

        return [
            operand.OperandRegister(chip, d + 16),
            operand.OperandImmediate(chip, K)
        ]

    @staticmethod
    def instruction_signature():
        return '0110 KKKK dddd KKKK'.replace(' ', '')

    # TODO: get_llil(), flags=SVNZ


class Instruction_SBRC(Instruction_SkipInstruction_Abstract):
    @staticmethod
    def instruction_signature():
        return '1111 110r rrrr 0bbb'.replace(' ', '')

    def _get_llil_condition(self, il):
        return il.compare_equal(
            1,
            il.test_bit(
                1,
                self._operands[0].llil_read(il),
                self._operands[1].llil_read(il)
            ),
            il.const(1, 0)
        )


class Instruction_SBRS(Instruction_SkipInstruction_Abstract):
    @staticmethod
    def instruction_signature():
        return '1111 111r rrrr 0bbb'.replace(' ', '')

    def _get_llil_condition(self, il):
        return il.compare_equal(
            1,
            il.test_bit(
                1,
                self._operands[0].llil_read(il),
                self._operands[1].llil_read(il)
            ),
            il.const(1, 1)
        )


class Instruction_SEC(Instruction):
    @staticmethod
    def instruction_signature():
        return '1001 0100 0000 1000'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_flag(
                'C',
                il.const(1, 1)
            )
        )


class Instruction_SEH(Instruction):
    @staticmethod
    def instruction_signature():
        return '1001 0100 0101 1000'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_flag(
                'H',
                il.const(1, 1)
            )
        )


class Instruction_SEI(Instruction):
    @staticmethod
    def instruction_signature():
        return '1001 0100 0111 1000'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_flag(
                'I',
                il.const(1, 1)
            )
        )


class Instruction_SEN(Instruction):
    @staticmethod
    def instruction_signature():
        return '1001 0100 0010 1000'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_flag(
                'N',
                il.const(1, 1)
            )
        )


class Instruction_SER(Instruction):
    @classmethod
    def args_to_operands(cls, chip, args):
        d = args['d']

        return [
            operand.OperandRegister(chip, d + 16)
        ]

    @staticmethod
    def instruction_signature():
        return '1110 1111 dddd 1111'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.const(1, 0xFF)
            )
        )


class Instruction_SES(Instruction):
    @staticmethod
    def instruction_signature():
        return '1001 0100 0100 1000'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_flag(
                'S',
                il.const(1, 1)
            )
        )


class Instruction_SET(Instruction):
    @staticmethod
    def instruction_signature():
        return '1001 0100 0110 1000'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_flag(
                'T',
                il.const(1, 1)
            )
        )


class Instruction_SEV(Instruction):
    @staticmethod
    def instruction_signature():
        return '1001 0100 0011 1000'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_flag(
                'V',
                il.const(1, 1)
            )
        )


class Instruction_SEZ(Instruction):
    @staticmethod
    def instruction_signature():
        return '1001 0100 0001 1000'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_flag(
                'Z',
                il.const(1, 1)
            )
        )


class Instruction_SLEEP(Instruction):
    @staticmethod
    def instruction_signature():
        return '1001 0101 1000 1000'.replace(' ', '')


class Instruction_SPM_I(Instruction):
    # TODO: Test RAMPZ.
    @classmethod
    def name(cls):
        return "spm"

    @staticmethod
    def instruction_signature():
        return '1001 0101 1110 1000'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.store(
                3,
                get_xyz_register(il, self._chip, 'Z'),
                il.or_expr(
                    2,
                    il.shift_left(
                        2,
                        il.reg(1, 'r1'),
                        il.const(1, 8)
                    ),
                    il.reg(1, 'r0')
                )
            )
        )


class Instruction_SPM_II(Instruction):
    # TODO: Add lifting.
    @classmethod
    def name(cls):
        return "spm[Z+]"

    @staticmethod
    def instruction_signature():
        return '1001 0101 1111 1000'.replace(' ', '')


class Instruction_ST_X_I(Instruction):
    @classmethod
    def name(cls):
        return "st"

    def get_instruction_text(self):
        tokens = [
            InstructionTextToken(
                InstructionTextTokenType.InstructionToken,
                self.name()
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ' '
            ),
            InstructionTextToken(
                InstructionTextTokenType.BeginMemoryOperandToken,
                '['
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                'X'
            ),
            InstructionTextToken(
                InstructionTextTokenType.EndMemoryOperandToken,
                ']'
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ', '
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                self.operands[0].symbolic_value
            ),
        ]

        return tokens

    @staticmethod
    def instruction_signature():
        return "1001 001r rrrr 1100".replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.store(
                1,
                il.add(
                    3,
                    il.const_pointer(3, RAM_SEGMENT_BEGIN),
                    get_xyz_register(il, self._chip, 'X')
                ),
                self._operands[0].llil_read(il),
            )
        )


class Instruction_ST_X_II(Instruction):
    @classmethod
    def name(cls):
        return "st"

    def get_instruction_text(self):
        tokens = [
            InstructionTextToken(
                InstructionTextTokenType.InstructionToken,
                self.name()
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ' '
            ),
            InstructionTextToken(
                InstructionTextTokenType.BeginMemoryOperandToken,
                '['
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                'X'
            ),
            InstructionTextToken(
                InstructionTextTokenType.StringToken,
                '+'
            ),
            InstructionTextToken(
                InstructionTextTokenType.EndMemoryOperandToken,
                ']'
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ', '
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                self.operands[0].symbolic_value
            ),
        ]

        return tokens

    @staticmethod
    def instruction_signature():
        return "1001 001r rrrr 1101".replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.store(
                1,
                il.add(
                    3,
                    il.const_pointer(3, RAM_SEGMENT_BEGIN),
                    get_xyz_register(il, self._chip, 'X')
                ),
                self._operands[0].llil_read(il),
            )
        )
        # X++
        il.append(
            il.set_reg(
                2,
                'X',
                il.add(
                    2,
                    il.reg(2, 'X'),
                    il.const(2, 1)
                )
            )
        )


class Instruction_ST_X_III(Instruction):
    @classmethod
    def name(cls):
        return "st"

    def get_instruction_text(self):
        tokens = [
            InstructionTextToken(
                InstructionTextTokenType.InstructionToken,
                self.name()
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ' '
            ),
            InstructionTextToken(
                InstructionTextTokenType.BeginMemoryOperandToken,
                '['
            ),
            InstructionTextToken(
                InstructionTextTokenType.StringToken,
                '-'
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                'X'
            ),
            InstructionTextToken(
                InstructionTextTokenType.EndMemoryOperandToken,
                ']'
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ', '
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                self.operands[0].symbolic_value
            ),
        ]

        return tokens

    @staticmethod
    def instruction_signature():
        return "1001 001r rrrr 1110".replace(' ', '')

    def get_llil(self, il):
        # --X
        il.append(
            il.set_reg(
                2,
                'X',
                il.sub(
                    2,
                    il.reg(2, 'X'),
                    il.const(2, 1)
                )
            )
        )

        il.append(
            il.store(
                1,
                il.add(
                    3,
                    il.const_pointer(3, RAM_SEGMENT_BEGIN),
                    get_xyz_register(il, self._chip, 'X')
                ),
                self._operands[0].llil_read(il),
            )
        )


class Instruction_ST_Y_I(Instruction):
    @classmethod
    def name(cls):
        return "st"

    def get_instruction_text(self):
        tokens = [
            InstructionTextToken(
                InstructionTextTokenType.InstructionToken,
                self.name()
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ' '
            ),
            InstructionTextToken(
                InstructionTextTokenType.BeginMemoryOperandToken,
                '['
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                'Y'
            ),
            InstructionTextToken(
                InstructionTextTokenType.EndMemoryOperandToken,
                ']'
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ', '
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                self.operands[0].symbolic_value
            ),
        ]

        return tokens

    @staticmethod
    def instruction_signature():
        return "1000 001r rrrr 1000".replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.store(
                1,
                il.add(
                    3,
                    il.const_pointer(3, RAM_SEGMENT_BEGIN),
                    get_xyz_register(il, self._chip, 'Y')
                ),
                self._operands[0].llil_read(il),
            )
        )


class Instruction_ST_Y_II(Instruction):
    @classmethod
    def name(cls):
        return "st"

    def get_instruction_text(self):
        tokens = [
            InstructionTextToken(
                InstructionTextTokenType.InstructionToken,
                self.name()
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ' '
            ),
            InstructionTextToken(
                InstructionTextTokenType.BeginMemoryOperandToken,
                '['
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                'Y'
            ),
            InstructionTextToken(
                InstructionTextTokenType.StringToken,
                '+'
            ),
            InstructionTextToken(
                InstructionTextTokenType.EndMemoryOperandToken,
                ']'
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ', '
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                self.operands[0].symbolic_value
            ),
        ]

        return tokens

    @staticmethod
    def instruction_signature():
        return "1001 001r rrrr 1001".replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.store(
                1,
                il.add(
                    3,
                    il.const_pointer(3, RAM_SEGMENT_BEGIN),
                    get_xyz_register(il, self._chip, 'Y')
                ),
                self._operands[0].llil_read(il),
            )
        )
        # Y++
        il.append(
            il.set_reg(
                2,
                'Y',
                il.add(
                    2,
                    il.reg(2, 'Y'),
                    il.const(2, 1)
                )
            )
        )


class Instruction_ST_Y_III(Instruction):
    @classmethod
    def name(cls):
        return "st"

    def get_instruction_text(self):
        tokens = [
            InstructionTextToken(
                InstructionTextTokenType.InstructionToken,
                self.name()
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ' '
            ),
            InstructionTextToken(
                InstructionTextTokenType.BeginMemoryOperandToken,
                '['
            ),
            InstructionTextToken(
                InstructionTextTokenType.StringToken,
                '-'
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                'Y'
            ),
            InstructionTextToken(
                InstructionTextTokenType.EndMemoryOperandToken,
                ']'
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ', '
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                self.operands[0].symbolic_value
            ),
        ]

        return tokens

    @staticmethod
    def instruction_signature():
        return "1001 001r rrrr 1010".replace(' ', '')

    def get_llil(self, il):
        # --Y
        il.append(
            il.set_reg(
                2,
                'Y',
                il.sub(
                    2,
                    il.reg(2, 'Y'),
                    il.const(2, 1)
                )
            )
        )

        il.append(
            il.store(
                1,
                il.add(
                    3,
                    il.const_pointer(3, RAM_SEGMENT_BEGIN),
                    get_xyz_register(il, self._chip, 'Y')
                ),
                self._operands[0].llil_read(il),
            )
        )


class Instruction_ST_Y_IV(Instruction):
    @classmethod
    def args_to_operands(cls, chip, args):
        r = args['r']
        q = args['q']

        return [
            operand.OperandRegister(chip, r),
            operand.OperandDirectAddress(chip, q + RAM_SEGMENT_BEGIN)
        ]

    @classmethod
    def name(cls):
        return "st"

    def get_instruction_text(self):
        tokens = [
            InstructionTextToken(
                InstructionTextTokenType.InstructionToken,
                self.name()
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ' '
            ),
            InstructionTextToken(
                InstructionTextTokenType.BeginMemoryOperandToken,
                '['
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                'Y'
            ),
            InstructionTextToken(
                InstructionTextTokenType.StringToken,
                ' + '
            ),
            InstructionTextToken(
                InstructionTextTokenType.IntegerToken,
                str(self.operands[1].immediate_value)
            ),
            InstructionTextToken(
                InstructionTextTokenType.EndMemoryOperandToken,
                ']'
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ', '
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                self.operands[0].symbolic_value
            ),
        ]

        return tokens

    @staticmethod
    def instruction_signature():
        return "10q0 qq1r rrrr 1qqq".replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.store(
                1,
                il.add(
                    3,
                    get_xyz_register(il, self._chip, 'Y'),
                    self._operands[1].llil_read(il)
                ),
                self._operands[0].llil_read(il),
            )
        )


class Instruction_ST_Z_I(Instruction):
    @classmethod
    def name(cls):
        return "st"

    def get_instruction_text(self):
        tokens = [
            InstructionTextToken(
                InstructionTextTokenType.InstructionToken,
                self.name()
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ' '
            ),
            InstructionTextToken(
                InstructionTextTokenType.BeginMemoryOperandToken,
                '['
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                'Z'
            ),
            InstructionTextToken(
                InstructionTextTokenType.EndMemoryOperandToken,
                ']'
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ', '
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                self.operands[0].symbolic_value
            ),
        ]

        return tokens

    @staticmethod
    def instruction_signature():
        return "1000 001r rrrr 0000".replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.store(
                1,
                il.add(
                    3,
                    il.const_pointer(3, RAM_SEGMENT_BEGIN),
                    get_xyz_register(il, self._chip, 'Z')
                ),
                self._operands[0].llil_read(il),
            )
        )


class Instruction_ST_Z_II(Instruction):
    @classmethod
    def name(cls):
        return "st"

    def get_instruction_text(self):
        tokens = [
            InstructionTextToken(
                InstructionTextTokenType.InstructionToken,
                self.name()
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ' '
            ),
            InstructionTextToken(
                InstructionTextTokenType.BeginMemoryOperandToken,
                '['
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                'Z'
            ),
            InstructionTextToken(
                InstructionTextTokenType.StringToken,
                '+'
            ),
            InstructionTextToken(
                InstructionTextTokenType.EndMemoryOperandToken,
                ']'
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ', '
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                self.operands[0].symbolic_value
            ),
        ]

        return tokens

    @staticmethod
    def instruction_signature():
        return "1001 001r rrrr 0001".replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.store(
                1,
                il.add(
                    3,
                    il.const_pointer(3, RAM_SEGMENT_BEGIN),
                    get_xyz_register(il, self._chip, 'Z')
                ),
                self._operands[0].llil_read(il),
            )
        )
        # Z++
        il.append(
            il.set_reg(
                2,
                'Z',
                il.add(
                    2,
                    il.reg(2, 'Z'),
                    il.const(2, 1)
                )
            )
        )


class Instruction_ST_Z_III(Instruction):
    @classmethod
    def name(cls):
        return "st"

    def get_instruction_text(self):
        tokens = [
            InstructionTextToken(
                InstructionTextTokenType.InstructionToken,
                self.name()
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ' '
            ),
            InstructionTextToken(
                InstructionTextTokenType.BeginMemoryOperandToken,
                '['
            ),
            InstructionTextToken(
                InstructionTextTokenType.StringToken,
                '-'
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                'Z'
            ),
            InstructionTextToken(
                InstructionTextTokenType.EndMemoryOperandToken,
                ']'
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ', '
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                self.operands[0].symbolic_value
            ),
        ]

        return tokens

    @staticmethod
    def instruction_signature():
        return "1001 001r rrrr 0010".replace(' ', '')

    def get_llil(self, il):
        # --Z
        il.append(
            il.set_reg(
                2,
                'Z',
                il.sub(
                    2,
                    il.reg(2, 'Z'),
                    il.const(2, 1)
                )
            )
        )

        il.append(
            il.store(
                1,
                il.add(
                    3,
                    il.const_pointer(3, RAM_SEGMENT_BEGIN),
                    get_xyz_register(il, self._chip, 'Z')
                ),
                self._operands[0].llil_read(il),
            )
        )


class Instruction_ST_Z_IV(Instruction):
    register_order = ['r', 'q']

    @classmethod
    def name(cls):
        return "st"

    def get_instruction_text(self):
        tokens = [
            InstructionTextToken(
                InstructionTextTokenType.InstructionToken,
                self.name()
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ' '
            ),
            InstructionTextToken(
                InstructionTextTokenType.BeginMemoryOperandToken,
                '['
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                'Z'
            ),
            InstructionTextToken(
                InstructionTextTokenType.StringToken,
                ' + '
            ),
            InstructionTextToken(
                InstructionTextTokenType.IntegerToken,
                str(self.operands[1].immediate_value)
            ),
            InstructionTextToken(
                InstructionTextTokenType.EndMemoryOperandToken,
                ']'
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ', '
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                self.operands[0].symbolic_value
            ),
        ]

        return tokens

    @staticmethod
    def instruction_signature():
        return "10q0 qq1r rrrr 0qqq".replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.store(
                1,
                il.add(
                    3,
                    il.const_pointer(3, RAM_SEGMENT_BEGIN),
                    il.add(
                        3,
                        get_xyz_register(il, self._chip, 'Z'),
                        self._operands[1].llil_read(il)
                    )
                ),
                self._operands[0].llil_read(il),
            )
        )


class Instruction_STS_32(Instruction):
    register_order = ['k', 'r']

    @classmethod
    def name(cls):
        return "sts"

    @staticmethod
    def instruction_signature():
        return '1001 001r rrrr 0000 kkkk kkkk kkkk kkkk'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.store(
                1,
                il.add(
                    3,
                    il.const_pointer(3, RAM_SEGMENT_BEGIN),
                    il.zero_extend(3, self._operands[0].llil_read(il))
                ),
                self._operands[1].llil_read(il),
            )
        )

    def get_instruction_text(self):
        tokens = [
            InstructionTextToken(
                InstructionTextTokenType.InstructionToken,
                self.name()
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ' '
            ),
            InstructionTextToken(
                InstructionTextTokenType.PossibleAddressToken,
                hex(self.operands[0].immediate_value + RAM_SEGMENT_BEGIN),
                value=self.operands[0].immediate_value + RAM_SEGMENT_BEGIN
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ', '
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                self.operands[1].symbolic_value
            ),
        ]

        return tokens


class Instruction_STS_16(Instruction):
    @classmethod
    def args_to_operands(cls, chip, args):
        k = args['k']
        r = args['r']

        return [
            operand.OperandImmediate(chip, k + 0x40),
            operand.OperandRegister(chip, r + 16)
        ]

    @classmethod
    def name(cls):
        return "sts"

    @staticmethod
    def instruction_signature():
        return '1010 1kkk rrrr kkkk'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.store(
                1,
                il.add(
                    3,
                    il.const_pointer(3, RAM_SEGMENT_BEGIN),
                    il.zero_extend(3, self._operands[1].llil_read(il))
                ),
                self._operands[0].llil_read(il),
            )
        )

    def get_instruction_text(self):
        tokens = [
            InstructionTextToken(
                InstructionTextTokenType.InstructionToken,
                self.name()
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ' '
            ),
            InstructionTextToken(
                InstructionTextTokenType.PossibleAddressToken,
                hex(self.operands[0].immediate_value + RAM_SEGMENT_BEGIN),
                value=self.operands[0].immediate_value + RAM_SEGMENT_BEGIN
            ),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ', '
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                self.operands[1].symbolic_value
            ),
        ]

        return tokens


class Instruction_SUB(Instruction):
    register_order = ['d', 'r']

    @staticmethod
    def instruction_signature():
        return '0001 10rd dddd rrrr'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.sub(
                    1,
                    self._operands[0].llil_read(il),
                    self._operands[1].llil_read(il),
                    flags='HSVNZC'
                )
            )
        )


class Instruction_SUBI(Instruction):
    @classmethod
    def args_to_operands(cls, chip, args):
        k = args['K']
        d = args['d']

        return [
            operand.OperandRegister(chip, d + 16),
            operand.OperandImmediate(chip, k)
        ]

    @staticmethod
    def instruction_signature():
        return '0101 KKKK dddd KKKK'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.sub(
                    1,
                    self._operands[0].llil_read(il),
                    self._operands[1].llil_read(il),
                    flags='HSVNZC'
                )
            )
        )


class Instruction_SWAP(Instruction):
    @staticmethod
    def instruction_signature():
        return '1001 010d dddd 0010'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.set_reg(
                1,
                self._operands[0].symbolic_value,
                il.rotate_right(
                    1,
                    self._operands[0].llil_read(il),
                    il.const(1, 4)
                )
            )
        )


class Instruction_TST(Instruction):
    register_order = ['d', 'r']

    def get_instruction_text(self):
        tokens = [
            InstructionTextToken(
                InstructionTextTokenType.InstructionToken, self.name()),
            InstructionTextToken(
                InstructionTextTokenType.OperandSeparatorToken,
                ' '
            ),
            InstructionTextToken(
                InstructionTextTokenType.RegisterToken,
                self.operands[0].symbolic_value
            ),
        ]

        return tokens

    @staticmethod
    def verify_args(args):
        return args['d'] == args['r']

    @staticmethod
    def instruction_signature():
        return '0010 00rd dddd rrrr'.replace(' ', '')

    def get_llil(self, il):
        il.append(
            il.and_expr(
                1,
                self._operands[0].llil_read(il),
                self._operands[0].llil_read(il),
                flags='SVNZ'
            )
        )


class Instruction_WDR(Instruction):
    @staticmethod
    def instruction_signature():
        return '1001 0101 1010 1000'.replace(' ', '')


class Instruction_XCH(Instruction):
    @staticmethod
    def instruction_signature():
        return '1001 001r rrrr 0100'.replace(' ', '')


ALL_INSTRUCTIONS = [
    # Easier-to-read aliases
    Instruction_CLR,
    Instruction_LSL,
    Instruction_ROL,
    Instruction_TST,

    Instruction_ADC,
    Instruction_ADD,
    Instruction_ADIW,
    Instruction_AND,
    Instruction_ANDI,
    Instruction_ASR,
    Instruction_BLD,
    Instruction_BRCC,
    Instruction_BRCS,
    Instruction_BREQ,
    Instruction_BRGE,
    Instruction_BRHC,
    Instruction_BRHS,
    Instruction_BRID,
    Instruction_BRIE,
    Instruction_BRLO,
    Instruction_BRLT,
    Instruction_BRMI,
    Instruction_BRNE,
    Instruction_BRPL,
    Instruction_BRSH,
    Instruction_BRTC,
    Instruction_BRTS,
    Instruction_BRVC,
    Instruction_BRVS,
    Instruction_BST,
    Instruction_CALL,
    Instruction_CBI,
    Instruction_CLC,
    Instruction_CLH,
    Instruction_CLI,
    Instruction_CLN,
    Instruction_CLS,
    Instruction_CLT,
    Instruction_CLV,
    Instruction_CLZ,
    Instruction_COM,
    Instruction_CP,
    Instruction_CPC,
    Instruction_CPI,
    Instruction_CPSE,
    Instruction_DEC,
    Instruction_EICALL,
    Instruction_EIJMP,
    Instruction_ELPM_I,
    Instruction_ELPM_II,
    Instruction_ELPM_III,
    Instruction_EOR,
    Instruction_FMUL,
    Instruction_FMULS,
    Instruction_FMULSU,
    Instruction_ICALL,
    Instruction_IJMP,
    Instruction_IN,
    Instruction_INC,
    Instruction_JMP,
    Instruction_LAC,
    Instruction_LAS,
    Instruction_LAT,
    Instruction_LD_X_I,
    Instruction_LD_X_II,
    Instruction_LD_X_III,
    Instruction_LD_Y_I,
    Instruction_LD_Y_II,
    Instruction_LD_Y_III,
    Instruction_LD_Y_IV,
    Instruction_LD_Z_I,
    Instruction_LD_Z_II,
    Instruction_LD_Z_III,
    Instruction_LD_Z_IV,
    Instruction_LDI,
    Instruction_LDS_16,
    Instruction_LDS_32,
    Instruction_LPM_I,
    Instruction_LPM_II,
    Instruction_LPM_III,
    Instruction_LSR,
    Instruction_MOV,
    Instruction_MOVW,
    Instruction_MUL,
    Instruction_MULS,
    Instruction_MULSU,
    Instruction_NEG,
    Instruction_NOP,
    Instruction_OR,
    Instruction_ORI,
    Instruction_OUT,
    Instruction_POP,
    Instruction_PUSH,
    Instruction_RCALL,
    Instruction_RET,
    Instruction_RETI,
    Instruction_RJMP,
    Instruction_ROR,
    Instruction_SBC,
    Instruction_SBCI,
    Instruction_SBI,
    Instruction_SBIC,
    Instruction_SBIS,
    Instruction_SBIW,
    Instruction_SBR,
    Instruction_SBRC,
    Instruction_SBRS,
    Instruction_SEC,
    Instruction_SEH,
    Instruction_SEI,
    Instruction_SEN,
    Instruction_SER,
    Instruction_SES,
    Instruction_SET,
    Instruction_SEV,
    Instruction_SEZ,
    Instruction_SLEEP,
    Instruction_SPM_I,
    Instruction_SPM_II,
    Instruction_ST_X_I,
    Instruction_ST_X_II,
    Instruction_ST_X_III,
    Instruction_ST_Y_I,
    Instruction_ST_Y_II,
    Instruction_ST_Y_III,
    Instruction_ST_Y_IV,
    Instruction_ST_Z_I,
    Instruction_ST_Z_II,
    Instruction_ST_Z_III,
    Instruction_ST_Z_IV,
    Instruction_STS_16,
    Instruction_STS_32,
    Instruction_SUB,
    Instruction_SUBI,
    Instruction_SWAP,
    Instruction_WDR,
    Instruction_XCH,
]

INSTRUCTIONS_BY_PREFIX = collections.defaultdict(
    lambda: collections.defaultdict(
        lambda: list()
    )
)


def populate_prefix_lookup_table():
    for ins in ALL_INSTRUCTIONS:
        prefix = []
        for c in ins.instruction_signature():
            if c in ['0', '1']:
                prefix.append(c)
            else:
                break

        if len(prefix) > 16:
            # Just in case.
            prefix = prefix[0:16]

        INSTRUCTIONS_BY_PREFIX[len(prefix)][int(''.join(prefix), 2)].append(ins)


# Caches all instructions that have been used by their first u16.
# Will contain 65k class references at max.
INSTRUCTION_CACHE = {}


def parse_instruction(chip, addr, data):
    """
    Tries to parse a single instruction.
    """
    if not data:
        return None

    # To reduce the overhead of parse_value we're parsing two u16 here and pass
    # it to parse_value.
    if len(data) >= 4:
        (u0, u1) = struct.unpack_from("<HH", data)
        w = 4
        v = u0 << 16 | u1
    else:
        u0 = struct.unpack_from("<H", data)[0]
        w = 2
        v = u0

    # Check cache.
    if u0 in INSTRUCTION_CACHE:
        ins = INSTRUCTION_CACHE[u0]
        return ins.parse_value(chip, addr, data, v if ins.length() == 4 else u0)

    for prefix_length, prefix_ins_list in INSTRUCTIONS_BY_PREFIX.items():
        prefix = u0 >> (16 - prefix_length)
        for ins in prefix_ins_list.get(prefix, []):
            if ins.length() > w:
                continue

            INSTRUCTION_CACHE[u0] = ins

            if ins.length() == 4:
                r = ins.parse_value(chip, addr, data, v)
            else:
                r = ins.parse_value(chip, addr, data, u0)

            if r:
                return r

    return None


populate_prefix_lookup_table()
