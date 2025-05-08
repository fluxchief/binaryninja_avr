from . import Chip


class IOTn88(Chip):
    """
    ATTiny88
    """

    CHIP_ALIASES = ["iotn88", "ATtiny88"]
    RAM_SIZE = 1024 * 16
    ROM_SIZE = 32 * 1024
    INTERRUPT_VECTOR_SIZE = 2

    IO_REGISTERS = {
        0x03: 'PINB',
        0x04: 'DDRB',
        0x05: 'PORTB',
        0x06: 'PINC',
        0x07: 'DDRC',
        0x08: 'PORTC',
        0x09: 'PIND',
        0x0A: 'DDRD',
        0x0B: 'PORTD',
        0x0C: 'PINA',
        0x0D: 'DDRA',
        0x0E: 'PORTA',
        # Res
        0x12: 'PORTCR',
        # Res
        0x15: 'TIFR0',
        0x16: 'TIFR1',
        # 0x17 - 0x1A res
        0x1B: 'PCIFR',
        0x1C: 'EIFR',
        0x1D: 'EIMSK',
        0x1E: 'GPIOR0',
        0x1F: 'EECR',
        0x20: 'EEDR',
        0x21: 'EEARL',
        # Res
        0x23: 'GTCCR',
        # Res
        0x25: 'TCCR0A',
        0x26: 'TCNT0',
        0x27: 'OCR0A',
        0x28: 'OCR0B',
        # Res
        0x2A: 'GPIOR1',
        0x2B: 'GPIOR2',
        0x2C: 'SPCR',
        0x2D: 'SPSR',
        0x2E: 'SPDR',
        # 0x2F reserved
        0x30: 'ACSR',
        0x31: 'DWDR',
        # Rserved
        0x33: 'SMCR',
        0x34: 'MCUSR',
        0x35: 'MCUCR',
        # 0x36 reserved
        0x37: 'SPMCSR',
        # 0x38 - 0x3C reserved
        0x3D: 'SPL',
        0x3E: 'SPH',
        0x3F: 'SREG',
    }

    INTERRUPT_VECTORS = [
        'RESET_vect',
        'INT0_vect',
        'INT1_vect',
        'PCINT0_vect',
        'PCINT1_vect',
        'PCINT2_vect',
        'PCINT3_vect',
        'WDT_vect',
        'TIMER1_CAPT_vect',
        'TIMER1_COMPA_vect',
        'TIMER1_COMPB_vect',
        'TIMER1_OVF_vect',
        'TIMER0_COMPA_vect',
        'TIMER0_COMPB_vect',
        'TIMER0_OVF_vect',
        'SPI_STC_vect',
        'ADC_vect',
        'EE_READY_vect',
        'ANALOG_COMP_vect',
        'TWI_vect',
    ]

    EXTENDED_IO_REGISTERS = {
        0x60: 'WDTCSR',
        0x61: 'CLKPR',
        0x64: 'PRR',
        0x66: 'OSCCAL',
        0x68: 'PCICR',
        0x69: 'EICRA',
        0x6A: 'PCMSK3',
        0x6B: 'PCMSK0',
        0x6C: 'PCMSK1',
        0x6D: 'PCMSK2',
        0x6E: 'TIMSK0',
        0x6F: 'TIMSK1',
        0x78: 'ADCL',
        0x79: 'ADCH',
        0x7A: 'ADCSRA',
        0x7B: 'ADCSRB',
        0x7C: 'ADMUX',
        0x7E: 'DIDR0',
        0x7F: 'DIDR1',
        0x80: 'TCCR1A',
        0x81: 'TCCR1B',
        0x82: 'TCCR1C',
        0x84: 'TCNT1L',
        0x85: 'TCNT1H',
        0x86: 'ICR1L',
        0x87: 'ICR1H',
        0x88: 'OCR1AL',
        0x89: 'OCR1AH',
        0x8A: 'OCR1BL',
        0x8B: 'OCR1BH',
        0xB8: 'TWBR',
        0xB9: 'TWSR',
        0xBA: 'TWAR',
        0xBB: 'TWDR',
        0xBC: 'TWCR',
        0xBD: 'TWAMR',
        0xBE: 'TWIHSR',
        0xBE: 'TWHSR',
    }
    @staticmethod
    def description():
        return "ATtiny88"

    @staticmethod
    def identifier():
        return __class__.__name__
