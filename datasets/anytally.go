package datasets

type AnyTally interface {

	// Erase
	Init()

	// Global Premodulo
	IsGlobalPremodulo() bool
	SetGlobalPremodulo(mod uint32)
	GetGlobalSaltPremodulo() [2]uint32
	GetGlobalPremodulo() uint32

	// Pre tallying
	GetCellDecision(position int, feature uint32) (bool, bool)
	SetCellDecision(position int, feature uint32, output bool)

	// Tallying
	AddToCorrect(feature uint32, vote int8, improvement bool)
	AddToImprove(feature uint32, vote int8)
	AddToMapping(feature uint16, output uint64)

	// Get Dataset at
	DatasetAt(n int) Dataset
	GetImprovementPossible() bool

	// Len
	Len() (ret int)
}

type TallyType byte

const PreTallyType TallyType = 2
const FinTallyType TallyType = 1

func NewAnyTally(typ TallyType) AnyTally {
	switch typ {
	case PreTallyType:
		t := &PreTally{}
		t.Init()
		return t
	case FinTallyType:
		t := &Tally{}
		t.Init()
		t.SetFinalization(true)
		return t
	default:
		return nil
	}
}
