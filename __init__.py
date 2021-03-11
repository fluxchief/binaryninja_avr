"""
Awesome binaryninja AVR disassembler/lifter plugin.
"""
import binascii
import struct

from . import instructions
from .instructions import RAM_SEGMENT_BEGIN


# TODO: This is ugly as hell. There should be some way to do this without
#       getting in some nasty dependency loop.
from .chips.iom16 import IOM16
from .chips.iom168 import IOM168
from .chips.iotn48 import IOTn48
from .chips.iotn88 import IOTn88
from .chips.iox128a4u import IOX128A4U
ALL_CHIPS = [
    IOM16,
    IOM168,
    IOTn48,
    IOTn88,
    IOX128A4U,
]


import binaryninja
from binaryninja import (
        BranchType, SegmentFlag, SectionSemantics, SymbolType,
        LowLevelILFlagCondition, FlagRole
)


class AVR(binaryninja.Architecture):
    name = 'AVR'
    address_size = 3
    default_int_size = 1
    # Instructions can only be 4 bytes in length MAX. However we need to have
    # the next instruction as well for some lifting reason, this is why we chose
    # twice the maximum value
    max_instr_length = 2 * 4
    instr_alignment = 2

    # Will be set in the binary view.
    chip = None

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

    # Kept as '0' most of the times
    global_regs = ['r1']

    stack_pointer = 'SP'
    flags = ['C', 'Z', 'N', 'V', 'S', 'H', 'T', 'I']
    flag_write_types = [
        '*',
        'HSVNZC',
        'HSVNZ',
        'SVNZC',
        'SVNZ',
        'ZC',
    ]

    flags_written_by_flag_write_type = {
        '*': ['C', 'Z', 'N', 'V', 'S', 'H', 'T', 'I'],
        'HSVNZC': ['H', 'S', 'V', 'N', 'Z', 'C'],
        'HSVNZ': ['H', 'S', 'V', 'N', 'Z'],
        'SVNZC': ['S', 'V', 'N', 'Z', 'C'],
        'SVNZ': ['S', 'V', 'N', 'Z'],
        'ZC': ['Z', 'C'],
    }

    flag_roles = {
        'C': FlagRole.CarryFlagRole,
        'Z': FlagRole.ZeroFlagRole,
        'N': FlagRole.NegativeSignFlagRole,
        'V': FlagRole.OverflowFlagRole,
        'S': FlagRole.SpecialFlagRole,        # (N ^ V)
        'H': FlagRole.HalfCarryFlagRole,
        'T': FlagRole.SpecialFlagRole,        # Transfer bit (BLD/BST)
        'I': FlagRole.SpecialFlagRole         # Global interrupt enable
    }

    flags_required_for_flag_condition = {
        LowLevelILFlagCondition.LLFC_E: ['Z'],              # Equal,      Z = 1
        LowLevelILFlagCondition.LLFC_NE: ['Z'],             # NEq,        Z = 0
        LowLevelILFlagCondition.LLFC_SLT: ['N', 'V'],       # < signed    N ^ V = 1
        LowLevelILFlagCondition.LLFC_ULT: ['C'],            # < usigned   C = 1
        LowLevelILFlagCondition.LLFC_SLE: ['N', 'V', 'Z'],  # <= signed   Z + (N ^ V) = 1
        LowLevelILFlagCondition.LLFC_ULE: ['C', 'Z'],       # <= unsiged  C + Z = 1
        LowLevelILFlagCondition.LLFC_SGE: ['N', 'V'],       # >= signed   N ^ V = 0
        LowLevelILFlagCondition.LLFC_UGE: ['C'],            # >= unsigned C = 0
        LowLevelILFlagCondition.LLFC_SGT: ['Z', 'N', 'V'],  # > signed    Z ? (N ^ V)
        LowLevelILFlagCondition.LLFC_UGT: ['C'],            # > unsigned  C = 0
        LowLevelILFlagCondition.LLFC_NEG: ['N'],            # is negative
        LowLevelILFlagCondition.LLFC_POS: ['N'],            # positive, obv inverted
        LowLevelILFlagCondition.LLFC_O: ['V'],              # overflow
        LowLevelILFlagCondition.LLFC_NO: ['V']              # no overflow
    }

    def _get_instruction(self, data, addr):
        return instructions.parse_instruction(self.chip, addr, data)

    def _is_conditional_branch(self, ins):
        return isinstance(ins, instructions.Instruction_BR_Abstract)

    def get_instruction_info(self, data, addr):
        nfo = binaryninja.InstructionInfo()
        ins = self._get_instruction(data, addr)
        if not ins:
            # Failsafe: Assume 2 bytes if we couldn't decode the instruction.
            # This should only happen if this is indeed an incorrect instruction
            # but for some reason BN tries to disassemble random data sometimes
            # and will show warnings if nfo.length == 0.
            binaryninja.log.log_warn(
                "Could not parse instruction @ 0x{:X}".format(
                    addr
                )
            )
            nfo.length = 2
            return nfo

        nfo.length = ins.length()

        if self._is_conditional_branch(ins):
            v = addr + ins.operands[0].immediate_value
            if v >= self.chip.ROM_SIZE:
                v -= self.chip.ROM_SIZE
            elif v < 0:
                v += self.chip.ROM_SIZE

            nfo.add_branch(
                BranchType.TrueBranch,
                v
            )
            nfo.add_branch(
                BranchType.FalseBranch,
                addr + 2
            )
        elif ins.__class__ in [
            instructions.Instruction_CPSE,
            instructions.Instruction_SBRC,
            instructions.Instruction_SBRS,
            instructions.Instruction_SBIC,
            instructions.Instruction_SBIS,
        ]:
            # TODO: This should skip a whole instruction but we don't know how
            # big the next instruction is (2 or 4 bytes).
            # Assume two bytes for now as it is pretty much always followed by a
            # rjmp
            nfo.add_branch(
                BranchType.TrueBranch,
                addr + 4
            )
            nfo.add_branch(
                BranchType.FalseBranch,
                addr + 2
            )
        elif isinstance(ins, instructions.Instruction_JMP):
            nfo.add_branch(
                BranchType.UnconditionalBranch,
                ins.operands[0].immediate_value
            )
        elif isinstance(ins, instructions.Instruction_CALL):
            nfo.add_branch(
                BranchType.CallDestination,
                ins.operands[0].immediate_value
            )
        elif (isinstance(ins, instructions.Instruction_RET) or
                isinstance(ins, instructions.Instruction_RETI)):
            nfo.add_branch(BranchType.FunctionReturn)
        elif (isinstance(ins, instructions.Instruction_RCALL)):
            v = addr + ins.operands[0].immediate_value
            if v >= self.chip.ROM_SIZE:
                v -= self.chip.ROM_SIZE
            elif v < 0:
                v += self.chip.ROM_SIZE

            nfo.add_branch(
                BranchType.CallDestination,
                v
            )
        elif (isinstance(ins, instructions.Instruction_RJMP)):
            v = addr + ins.operands[0].immediate_value
            if v >= self.chip.ROM_SIZE:
                v -= self.chip.ROM_SIZE
            elif v < 0:
                v += self.chip.ROM_SIZE

            nfo.add_branch(
                BranchType.UnconditionalBranch,
                v
            )
        elif (isinstance(ins, instructions.Instruction_ICALL) or
                isinstance(ins, instructions.Instruction_EICALL) or
                isinstance(ins, instructions.Instruction_IJMP) or
                isinstance(ins, instructions.Instruction_EIJMP)):
            nfo.add_branch(BranchType.IndirectBranch)
        else:
            # TODO: Doublecheck that there are no more controlflow modifying
            # operations.
            pass

        return nfo

    def get_instruction_text(self, data, addr):
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

    def get_instruction_low_level_il(self, data, addr, il):
        ins = self._get_instruction(data, addr)
        if ins:
            ins.get_llil(il)
            return ins.length()
        else:
            binaryninja.log_warn(
                "Could not parse instruction @ 0x{:08X}".format(
                    addr
                )
            )
            il.append(il.no_ret())
            return 0

    def is_never_branch_patch_available(self, data, addr):
        ins = self._get_instruction(data, addr)
        return self._is_conditional_branch(ins)

    def is_always_branch_patch_available(self, data, addr):
        ins = self._get_instruction(data, addr)
        return self._is_conditional_branch(ins)

    def always_branch(self, data, addr):
        ins = self._get_instruction(data, addr)
        dst = ins._operands[0]
        v = (dst.immediate_value - 2) / 2
        v = (v & 0xFFF) | 0xc000
        return struct.pack('<H', v)

    def never_branch(self, data, addr):
        return "\x00\x00"

    def convert_to_nop(self, data, addr):
        return "\x00\x00"

    """
    def get_flag_write_low_level_il(self, op, size, write_type, flag, operands, il):
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
        chip = ALL_CHIPS[
            binaryninja.interaction.get_choice_input(
                "Target chip",
                "Select target chip",
                [
                    ' '.join(c.CHIP_ALIASES)
                    for c in ALL_CHIPS
                ],
            )
        ]

        # Setting this somewhat globally.
        # TODO: Figure out if there is a way to have separate instances for each
        #       open window / tab.
        AVR.chip = chip()

        if len(self.raw) > AVR.chip.ROM_SIZE:
            binaryninja.log.log_error("AVR: Rom too big for this chip")
            return False

        self.platform = binaryninja.Architecture[AVR.name].standalone_platform
        self.arch = binaryninja.Architecture[AVR.name]

        self.add_auto_segment(
            0, AVR.chip.ROM_SIZE,
            0, len(self.raw),
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentExecutable
        )
        self.add_auto_section("ROM", 0, AVR.chip.ROM_SIZE,
                              SectionSemantics.ReadOnlyCodeSectionSemantics)

        # Register / IO / Extended IO.
        memory_mapped_registers_size = max([
            a for a, _ in AVR.chip.all_registers.items()
        ]) + 1

        self.add_auto_segment(
            RAM_SEGMENT_BEGIN, memory_mapped_registers_size,
            RAM_SEGMENT_BEGIN, 0,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentWritable
        )
        self.add_auto_section("Memory mapped registers (IO)", RAM_SEGMENT_BEGIN,
                              memory_mapped_registers_size,
                              SectionSemantics.ReadWriteDataSectionSemantics)

        # Make types.
        type_u8 = self.parse_type_string("uint8_t")[0]

        # All registers.
        for addr, name in AVR.chip.all_registers.items():
            self.define_data_var(RAM_SEGMENT_BEGIN + addr, type_u8)
            self.define_auto_symbol(binaryninja.types.Symbol(
                SymbolType.DataSymbol,
                RAM_SEGMENT_BEGIN + addr,
                name
            ))

        # Actual RAM.
        self.add_auto_segment(
            RAM_SEGMENT_BEGIN + AVR.chip.RAM_STARTS_AT, AVR.chip.RAM_SIZE,
            RAM_SEGMENT_BEGIN, 0,
            SegmentFlag.SegmentReadable | SegmentFlag.SegmentWritable
        )
        self.add_auto_section("RAM", RAM_SEGMENT_BEGIN + AVR.chip.RAM_STARTS_AT,
                              AVR.chip.RAM_SIZE,
                              SectionSemantics.ReadWriteDataSectionSemantics)

        # Create ISR table.
        for i, v in enumerate(AVR.chip.INTERRUPT_VECTORS):
            isr_addr = i * AVR.chip.INTERRUPT_VECTOR_SIZE
            if not self.get_function_at(isr_addr):
                self.add_function(isr_addr)

            f = self.get_function_at(isr_addr)
            f.name = "j_{}".format(v)
            try:
                jmp_target = f.lifted_il[0].operands[0].operands[0]
            except:
                binaryninja.log.log_error(
                    "Failed to parse jump target at 0x{:X} - incorrect chip?"
                    .format(isr_addr)
                )
                jmp_target = None

            if jmp_target:
                if not self.get_function_at(jmp_target):
                    self.add_function(jmp_target)

                if self.get_function_at(jmp_target).name == "sub_{:x}".format(jmp_target):
                    self.get_function_at(jmp_target).name = v

        self.add_entry_point(0)
        return True

    def is_executable(self):
        return True

    def get_entry_point(self):
        return 0

    @classmethod
    def is_valid_for_data(self, data):
        return True


AVR.register()
arch = binaryninja.Architecture[AVR.name]
arch.register_calling_convention(DefaultCallingConvention(arch, 'default'))

arch.standalone_platform.default_calling_convention = arch.calling_conventions['default']
AVRBinaryView.register()

