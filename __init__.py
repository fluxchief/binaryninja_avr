"""
Awesome binaryninja AVR disassembler/lifter plugin.
"""
import binascii

# Load all chips
import binaryninja_avr.chips.iom16
import binaryninja_avr.chips.iotn48
import binaryninja_avr.chips.iotn88
import binaryninja_avr.chips.iox256a4u

import binaryninja_avr.instructions
from binaryninja_avr.instructions import RAM_SEGMENT_BEGIN
import binaryninja_avr.operand

import binaryninja


class AVR(binaryninja.Architecture):
    name = 'AVR'
    address_size = 2
    default_int_size = 1
    # Instructions can only be 4 bytes in length MAX. However we need to have
    # the next instruction as well for some lifting reason, this is why we chose
    # twice the maximum value
    max_instr_length = 2 * 4

    # Ideally we want some select box here, no support in BN for this yet.

    # chip = binaryninja_avr.chips.iom16.IOM16()
    # chip = binaryninja_avr.chips.iotn48.IOTn48()
    # chip = binaryninja_avr.chips.iotn88.IOTn88()
    chip = binaryninja_avr.chips.iox256a4u.IOX256A4U()

    regs = {
        'r0': binaryninja.RegisterInfo('r0', 1),
        'r1': binaryninja.RegisterInfo('r1', 1),
        'r2': binaryninja.RegisterInfo('r2', 1),
        'r3': binaryninja.RegisterInfo('r3', 1),
        'r4': binaryninja.RegisterInfo('r4', 1),
        'r5': binaryninja.RegisterInfo('r5', 1),
        'r6': binaryninja.RegisterInfo('r6', 1),
        'r7': binaryninja.RegisterInfo('r7', 1),
        'r8': binaryninja.RegisterInfo('r8', 1),
        'r9': binaryninja.RegisterInfo('r9', 1),
        'r10': binaryninja.RegisterInfo('r10', 1),
        'r11': binaryninja.RegisterInfo('r11', 1),
        'r12': binaryninja.RegisterInfo('r12', 1),
        'r13': binaryninja.RegisterInfo('r13', 1),
        'r14': binaryninja.RegisterInfo('r14', 1),
        'r15': binaryninja.RegisterInfo('r15', 1),
        'r16': binaryninja.RegisterInfo('r16', 1),
        'r17': binaryninja.RegisterInfo('r17', 1),
        'r18': binaryninja.RegisterInfo('r18', 1),
        'r19': binaryninja.RegisterInfo('r19', 1),
        'r20': binaryninja.RegisterInfo('r20', 1),
        'r21': binaryninja.RegisterInfo('r21', 1),
        'r22': binaryninja.RegisterInfo('r22', 1),
        'r23': binaryninja.RegisterInfo('r23', 1),
        'r24': binaryninja.RegisterInfo('r24', 1),
        'r25': binaryninja.RegisterInfo('r25', 1),
        'r26': binaryninja.RegisterInfo('r26', 1),
        'r27': binaryninja.RegisterInfo('r27', 1),
        'r28': binaryninja.RegisterInfo('r28', 1),
        'r29': binaryninja.RegisterInfo('r29', 1),
        'r30': binaryninja.RegisterInfo('r30', 1),
        'r31': binaryninja.RegisterInfo('r31', 1),
        'SP': binaryninja.RegisterInfo('SP', 2),
    }

    stack_pointer = 'SP'
    flags = ['C', 'Z', 'N', 'V', 'S', 'H', 'T', 'I']
    flag_write_types = [
        '',  # https://github.com/Vector35/binaryninja-api/issues/513
        '*',
        'HSVNZC',
        'HSVNZ',
        'SVNZC',
        'SVNZ',
        'ZC',
    ]

    flags_written_by_flag_write_type = {
        '': [],
        '*': ['C', 'Z', 'N', 'V', 'S', 'H', 'T', 'I'],
        'HSVNZC': ['H', 'S', 'V', 'N', 'Z', 'C'],
        'HSVNZ': ['H', 'S', 'V', 'N', 'Z'],
        'SVNZC': ['S', 'V', 'N', 'Z', 'C'],
        'SVNZ': ['S', 'V', 'N', 'Z'],
        'ZC': ['Z', 'C'],
    }

    flag_roles = {
        'C': binaryninja.enums.FlagRole.CarryFlagRole,
        'Z': binaryninja.enums.FlagRole.ZeroFlagRole,
        'N': binaryninja.enums.FlagRole.NegativeSignFlagRole,
        'V': binaryninja.enums.FlagRole.OverflowFlagRole,
        'S': binaryninja.enums.FlagRole.SpecialFlagRole,        # (N ^ V)
        'H': binaryninja.enums.FlagRole.HalfCarryFlagRole,
        'T': binaryninja.enums.FlagRole.SpecialFlagRole,        # Transfer bit (BLD/BST)
        'I': binaryninja.enums.FlagRole.SpecialFlagRole         # Global interrupt enable
    }

    flags_required_for_flag_condition = {
        binaryninja.enums.LowLevelILFlagCondition.LLFC_E: ['Z'],              # Equal,      Z = 1
        binaryninja.enums.LowLevelILFlagCondition.LLFC_NE: ['Z'],             # NEq,        Z = 0
        binaryninja.enums.LowLevelILFlagCondition.LLFC_SLT: ['N', 'V'],       # < signed    N ^ V = 1
        binaryninja.enums.LowLevelILFlagCondition.LLFC_ULT: ['C'],            # < usigned   C = 1
        binaryninja.enums.LowLevelILFlagCondition.LLFC_SLE: ['N', 'V', 'Z'],  # <= signed   Z + (N ^ V) = 1
        binaryninja.enums.LowLevelILFlagCondition.LLFC_ULE: ['C', 'Z'],       # <= unsiged  C + Z = 1
        binaryninja.enums.LowLevelILFlagCondition.LLFC_SGE: ['N', 'V'],       # >= signed   N ^ V = 0
        binaryninja.enums.LowLevelILFlagCondition.LLFC_UGE: ['C'],            # >= unsigned C = 0
        binaryninja.enums.LowLevelILFlagCondition.LLFC_SGT: ['Z', 'N', 'V'],  # > signed    Z ? (N ^ V)
        binaryninja.enums.LowLevelILFlagCondition.LLFC_UGT: ['C'],            # > unsigned  C = 0
        binaryninja.enums.LowLevelILFlagCondition.LLFC_NEG: ['N'],            # is negative
        binaryninja.enums.LowLevelILFlagCondition.LLFC_POS: ['N'],            # positive, obv inverted
        binaryninja.enums.LowLevelILFlagCondition.LLFC_O: ['V'],              # overflow
        binaryninja.enums.LowLevelILFlagCondition.LLFC_NO: ['V']              # no overflow
    }

    def _get_instruction(self, data, addr):
        return binaryninja_avr.instructions.parse_instruction(self.chip, addr, data)

    def perform_get_instruction_info(self, data, addr):
        nfo = binaryninja.InstructionInfo()
        ins = self._get_instruction(data, addr)
        if not ins:
            # Failsafe: Assume 2 bytes if we couldn't decode the instruction.
            binaryninja.log.log_warn(
                "Could not get instruction @ 0x{:X}, assuming len=2".format(
                    addr
                )
            )
            nfo.length = 2
            return nfo

        nfo.length = ins.length()

        if ins.__class__ in [
            binaryninja_avr.instructions.Instruction_BRCC,
            binaryninja_avr.instructions.Instruction_BRCS,
            binaryninja_avr.instructions.Instruction_BREQ,
            binaryninja_avr.instructions.Instruction_BRGE,
            binaryninja_avr.instructions.Instruction_BRHC,
            binaryninja_avr.instructions.Instruction_BRHS,
            binaryninja_avr.instructions.Instruction_BRID,
            binaryninja_avr.instructions.Instruction_BRIE,
            binaryninja_avr.instructions.Instruction_BRLO,
            binaryninja_avr.instructions.Instruction_BRLT,
            binaryninja_avr.instructions.Instruction_BRMI,
            binaryninja_avr.instructions.Instruction_BRNE,
            binaryninja_avr.instructions.Instruction_BRPL,
            binaryninja_avr.instructions.Instruction_BRSH,
            binaryninja_avr.instructions.Instruction_BRTC,
            binaryninja_avr.instructions.Instruction_BRTS,
            binaryninja_avr.instructions.Instruction_BRVC,
            binaryninja_avr.instructions.Instruction_BRVS,
        ]:
            v = addr + ins.operands[0].immediate_value
            if v >= self.chip.ROM_SIZE:
                v -= self.chip.ROM_SIZE
            elif v < 0:
                v += self.chip.ROM_SIZE

            nfo.add_branch(
                binaryninja.BranchType.TrueBranch,
                v
            )
            nfo.add_branch(
                binaryninja.BranchType.FalseBranch,
                addr + 2
            )
        elif ins.__class__ in [
            binaryninja_avr.instructions.Instruction_CPSE,
            binaryninja_avr.instructions.Instruction_SBRC,
            binaryninja_avr.instructions.Instruction_SBRS,
            binaryninja_avr.instructions.Instruction_SBIC,
            binaryninja_avr.instructions.Instruction_SBIS,
        ]:
            # TODO: This should skip a whole instruction but we don't know how
            # big the next instruction is (2 or 4 bytes).
            # Assume two bytes for now as it is pretty much always followed by a
            # rjmp
            nfo.add_branch(
                binaryninja.BranchType.TrueBranch,
                addr + 4
            )
            nfo.add_branch(
                binaryninja.BranchType.FalseBranch,
                addr + 2
            )
        elif isinstance(ins, binaryninja_avr.instructions.Instruction_JMP):
            nfo.add_branch(
                binaryninja.BranchType.UnconditionalBranch,
                ins.operands[0].immediate_value
            )
        elif isinstance(ins, binaryninja_avr.instructions.Instruction_CALL):
            nfo.add_branch(
                binaryninja.BranchType.CallDestination,
                ins.operands[0].immediate_value
            )
        elif (isinstance(ins, binaryninja_avr.instructions.Instruction_RET) or
                isinstance(ins, binaryninja_avr.instructions.Instruction_RETI)):
            nfo.add_branch(binaryninja.BranchType.FunctionReturn)
        elif (isinstance(ins, binaryninja_avr.instructions.Instruction_RCALL)):
            v = addr + ins.operands[0].immediate_value
            if v >= self.chip.ROM_SIZE:
                v -= self.chip.ROM_SIZE
            elif v < 0:
                v += self.chip.ROM_SIZE

            nfo.add_branch(
                binaryninja.BranchType.CallDestination,
                v
            )
        elif (isinstance(ins, binaryninja_avr.instructions.Instruction_RJMP)):
            v = addr + ins.operands[0].immediate_value
            if v >= self.chip.ROM_SIZE:
                v -= self.chip.ROM_SIZE
            elif v < 0:
                v += self.chip.ROM_SIZE

            nfo.add_branch(
                binaryninja.BranchType.UnconditionalBranch,
                v
            )
        elif (isinstance(ins, binaryninja_avr.instructions.Instruction_ICALL) or
                isinstance(ins, binaryninja_avr.instructions.Instruction_EICALL)):
            nfo.add_branch(binaryninja.BranchType.IndirectBranch)
        else:
            # TODO: Doublecheck that there are no more controlflow modifying
            # operations.
            pass

        return nfo

    def perform_get_instruction_text(self, data, addr):
        ins = self._get_instruction(data, addr)
        if not ins:
            return [
                binaryninja.InstructionTextToken(
                    binaryninja.InstructionTextTokenType.InstructionToken,
                    "Unsupported ({})".format(
                        binascii.hexlify(data)
                    )
                )
            ], 2

        return ins.get_instruction_text(), ins.length()

    def perform_get_instruction_low_level_il(self, data, addr, il):
        ins = self._get_instruction(data, addr)
        if ins:
            ins.get_llil(il)
            return ins.length()
        else:
            il.append(il.unimplemented())
            return 2

    """
    def perform_get_flag_write_low_level_il(self, op, size, write_type, flag, operands, il):
        return
    """


class DefaultCallingConvention(binaryninja.CallingConvention):
    name = 'default'
    int_arg_regs = ['r22', 'r23', 'r24', 'r25']
    int_return_reg = 'r30'
    high_int_return_reg = 'r31'


class AVRBinaryView(binaryninja.BinaryView):
    name = 'AVR'
    long_name = 'Atmel AVR'

    def __init__(self, data):
        binaryninja.BinaryView.__init__(self, file_metadata=data.file, parent_view=data)
        self.raw = data

    def __undefine_symbol_if_defined(self, addr):
        s = self.get_symbol_at(addr)
        if s:
            self.undefine_auto_symbol(s)

    def init(self):
        if len(self.raw) > AVR.chip.ROM_SIZE:
            binaryninja.log.log_error("AVR: Rom too big for this chip")
            return False

        self.platform = binaryninja.Architecture[AVR.name].standalone_platform
        self.arch = binaryninja.Architecture[AVR.name]

        self.add_auto_segment(
            0, AVR.chip.ROM_SIZE,
            0, len(self.raw),
            binaryninja.SegmentFlag.SegmentReadable | binaryninja.SegmentFlag.SegmentExecutable
        )

        # Register / IO / Extended IO.
        self.add_auto_segment(
            RAM_SEGMENT_BEGIN, max([a for a, _ in AVR.chip.all_registers.items()]) + 1,
            RAM_SEGMENT_BEGIN, 0,
            binaryninja.SegmentFlag.SegmentReadable | binaryninja.SegmentFlag.SegmentWritable
        )

        # Make types.
        type_u8 = self.parse_type_string("uint8_t")[0]

        # All registers.
        for addr, name in AVR.chip.all_registers.items():
            self.define_data_var(RAM_SEGMENT_BEGIN + addr, type_u8)
            self.define_auto_symbol(binaryninja.types.Symbol(
                binaryninja.enums.SymbolType.DataSymbol,
                RAM_SEGMENT_BEGIN + addr,
                name
            ))

        # Actual RAM.
        self.add_auto_segment(
            RAM_SEGMENT_BEGIN + AVR.chip.RAM_STARTS_AT, AVR.chip.RAM_SIZE,
            RAM_SEGMENT_BEGIN, 0,
            binaryninja.SegmentFlag.SegmentReadable | binaryninja.SegmentFlag.SegmentWritable
        )

        # Create ISR table.
        for i, v in enumerate(AVR.chip.INTERRUPT_VECTORS):
            isr_addr = i * AVR.chip.INTERRUPT_VECTOR_SIZE
            if not self.get_function_at(isr_addr):
                self.add_function(isr_addr)

            f = self.get_function_at(isr_addr)
            f.name = "j_{}".format(v)
            jmp_target = f.medium_level_il[0].operands[0].operands[0]
            if jmp_target:
                if not self.get_function_at(jmp_target):
                    self.add_function(jmp_target)

                if self.get_function_at(jmp_target).name == "sub_{:x}".format(jmp_target):
                    self.get_function_at(jmp_target).name = v

        self.add_entry_point(0)
        return True

    def perform_is_executable(self):
        return True

    def perform_get_entry_point(self):
        return 0

    @classmethod
    def is_valid_for_data(self, data):
        return True


AVR.register()
AVRBinaryView.register()

# Uhm, copy paste?
arch = binaryninja.Architecture[AVR.name]
cc = DefaultCallingConvention(arch, name='default')
arch.register_calling_convention(cc)
arch.standalone_platform.default_calling_convention = arch.calling_conventions[
    'default']
