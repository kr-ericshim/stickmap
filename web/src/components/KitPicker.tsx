import { FAMILY, type Inst, type SoundName } from '../api'
import { FAMILY_COLOR } from './GrooveMap'

// The kit seen from the drummer's throne, so a stroke seen in the video is picked where it was played.
type Piece = { name: string; inst: Inst; shape: 'circle' | 'ellipse'; x: number; y: number; r: number; ry?: number }

const PIECES: Piece[] = [
  { name: '크래시', inst: 'cymbal', shape: 'circle', x: 62, y: 34, r: 26 },
  { name: '라이드', inst: 'ride', shape: 'circle', x: 172, y: 46, r: 32 },
  { name: '하이햇', inst: 'hihat', shape: 'circle', x: 30, y: 96, r: 20 },
  { name: '하이 탐', inst: 'tom_high', shape: 'circle', x: 88, y: 74, r: 15 },
  { name: '미들 탐', inst: 'tom', shape: 'circle', x: 124, y: 74, r: 17 },
  { name: '스네어', inst: 'snare', shape: 'circle', x: 70, y: 120, r: 19 },
  { name: '로우 탐', inst: 'tom_low', shape: 'circle', x: 164, y: 124, r: 23 },
  { name: '킥', inst: 'kick', shape: 'ellipse', x: 116, y: 130, r: 26, ry: 14 },
]

// Ways of playing one piece: picked after the piece itself.
const STROKES: Partial<Record<Inst, { inst: Inst; label: string }[]>> = {
  hihat: [
    { inst: 'hihat', label: '닫힘' },
    { inst: 'hihat_open', label: '열림' },
    { inst: 'hihat_pedal', label: '페달' },
  ],
  snare: [
    { inst: 'snare', label: '일반' },
    { inst: 'sidestick', label: '사이드 스틱' },
  ],
  ride: [
    { inst: 'ride', label: '보우' },
    { inst: 'ride_bell', label: '벨' },
  ],
}

const PIECE_OF: Record<Inst, Inst> = {
  kick: 'kick',
  snare: 'snare',
  sidestick: 'snare',
  hihat: 'hihat',
  hihat_open: 'hihat',
  hihat_pedal: 'hihat',
  tom_high: 'tom_high',
  tom: 'tom',
  tom_low: 'tom_low',
  ride: 'ride',
  ride_bell: 'ride',
  cymbal: 'cymbal',
}

export const INST_LABEL: Record<Inst, string> = {
  kick: '킥',
  snare: '스네어',
  sidestick: '사이드 스틱',
  hihat: '하이햇',
  hihat_open: '오픈 하이햇',
  hihat_pedal: '페달 하이햇',
  tom_high: '하이 탐',
  tom: '미들 탐',
  tom_low: '로우 탐',
  ride: '라이드',
  ride_bell: '라이드 벨',
  cymbal: '크래시',
}

type Props = {
  value: SoundName | null
  onChange: (name: SoundName) => void
  disabled?: boolean
}

export function KitPicker({ value, onChange, disabled }: Props) {
  const piece = value && value !== 'ignore' ? PIECE_OF[value] : null
  const strokes = piece ? STROKES[piece] : undefined

  return (
    <div className="grid content-start gap-2">
      <svg viewBox="0 0 220 176" className="mx-auto w-full max-w-[300px]" role="group" aria-label="위에서 본 드럼 킷">
        {PIECES.map((p) => {
          const chosen = p.inst === piece
          const common = {
            role: 'button',
            tabIndex: disabled ? -1 : 0,
            'aria-label': p.name,
            'aria-pressed': chosen,
            onClick: () => !disabled && onChange(p.inst),
            onKeyDown: (event: React.KeyboardEvent) => {
              if (!disabled && (event.key === 'Enter' || event.key === ' ')) {
                event.preventDefault()
                onChange(p.inst)
              }
            },
            className: `cursor-pointer outline-none transition-[fill] duration-150 focus-visible:stroke-brass ${
              disabled ? 'cursor-default opacity-50' : 'hover:stroke-ink'
            }`,
            fill: chosen ? FAMILY_COLOR[FAMILY[p.inst]] : 'transparent',
            stroke: chosen ? FAMILY_COLOR[FAMILY[p.inst]] : 'var(--color-mute)',
            strokeWidth: 1.4,
          }
          return p.shape === 'circle' ? (
            <circle key={p.name} cx={p.x} cy={p.y} r={p.r} {...common} />
          ) : (
            <ellipse key={p.name} cx={p.x} cy={p.y} rx={p.r} ry={p.ry} {...common} />
          )
        })}
      </svg>
      <div className="flex min-h-7 flex-wrap items-center gap-1.5">
        {strokes?.map((stroke) => (
          <button
            key={stroke.inst}
            disabled={disabled}
            aria-pressed={value === stroke.inst}
            onClick={() => onChange(stroke.inst)}
            className={`rounded-full px-3 py-1 text-xs font-medium disabled:opacity-50 ${
              value === stroke.inst ? 'bg-ink text-paper' : 'bg-ink/6 text-mute hover:text-ink'
            }`}
          >
            {stroke.label}
          </button>
        ))}
      </div>
      <div className="flex items-center justify-between text-xs">
        <span className="text-mute">
          {value === 'ignore' ? '악보에서 뺌' : value ? INST_LABEL[value] : '드럼을 눌러 고르세요'}
        </span>
        <button
          disabled={disabled}
          aria-pressed={value === 'ignore'}
          onClick={() => onChange('ignore')}
          className={`rounded-full px-3 py-1 font-medium disabled:opacity-50 ${
            value === 'ignore' ? 'bg-ink text-paper' : 'bg-ink/6 text-mute hover:text-ink'
          }`}
        >
          무시
        </button>
      </div>
    </div>
  )
}
