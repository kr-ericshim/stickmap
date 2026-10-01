import { useEffect, useState } from 'react'
import { api, FAMILY, type Family, type SoundGroup, type SoundName, type Song } from '../api'
import { formatClock } from '../timeline'
import { ClipMonitor } from './ClipMonitor'
import { FAMILY_COLOR } from './GrooveMap'
import { INST_LABEL, KitPicker } from './KitPicker'

const FAMILY_LABEL: Record<Family, string> = { kick: '킥', snare: '스네어', tom: '탐', hihat: '하이햇', cymbal: '심벌' }

type Props = {
  song: Song
  onChange: (song: Song) => void
  onError: (message: string) => void
}

// The song's sounds, one card each: watch a stroke slowly, hear it, and say which drum it was.
// Naming is how a kit the detector does not know gets written right, so nothing is renamed by itself.
export function SoundPanel({ song, onChange, onError }: Props) {
  const [groups, setGroups] = useState<SoundGroup[] | null>(null)
  const [open, setOpen] = useState<string | null>(null)
  const [example, setExample] = useState(0)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    let cancelled = false
    setGroups(null)
    api
      .sounds(song.slug)
      .then((map) => {
        if (cancelled) return
        setGroups(map.groups)
        setOpen(map.groups.find((g) => !g.name)?.id ?? map.groups[0]?.id ?? null)
      })
      .catch((error: Error) => !cancelled && onError(error.message))
    return () => {
      cancelled = true
    }
  }, [song.slug, onError])

  const save = async (id: string, name: SoundName | null) => {
    if (!groups) return
    const names: Record<string, SoundName> = {}
    for (const g of groups) {
      const value = g.id === id ? name : g.name
      if (value) names[g.id] = value
    }
    setBusy(true)
    try {
      onChange(await api.nameSounds(song.slug, names))
      setGroups(groups.map((g) => (g.id === id ? { ...g, name } : g)))
    } catch (error) {
      onError(error instanceof Error ? error.message : String(error))
    } finally {
      setBusy(false)
    }
  }

  if (!groups) {
    return <p className="px-6 py-10 text-center text-sm text-mute">곡의 소리를 나누는 중…</p>
  }
  if (!groups.length) {
    return <p className="px-6 py-10 text-center text-sm text-mute">나눌 소리가 없습니다.</p>
  }

  const named = groups.filter((g) => g.name).length

  return (
    <div className="grid gap-3 px-5 py-5">
      <div className="grid gap-2">
        <div className="flex items-baseline gap-3">
          <h2 className="font-display text-xl font-bold whitespace-nowrap tracking-tight [font-stretch:85%]">
            이 곡의 소리
          </h2>
          <span className="font-mono text-xs text-mute">
            {groups.length}개 중 {named}개 확인
          </span>
        </div>
        <span className="flex flex-wrap gap-1" aria-hidden="true">
          {groups.map((g) => (
            <Dot key={g.id} group={g} />
          ))}
        </span>
      </div>
      <p className="text-xs leading-relaxed text-mute">
        소리를 느리게 보고 들은 뒤, 친 드럼을 킷 그림에서 고르세요. 고른 이름은 그 소리 전체에 적용되고 악보가 다시
        그려집니다.
      </p>

      {groups.map((group) => {
        const expanded = group.id === open
        const current = group.examples[Math.min(example, group.examples.length - 1)]
        return (
          <section
            key={group.id}
            className={`rounded-lg bg-paper ${expanded ? 'shadow-[0_0_0_1px_var(--color-line),0_8px_24px_-16px_rgba(20,54,63,0.4)]' : 'shadow-[0_0_0_1px_var(--color-line)]'}`}
          >
            <button
              onClick={() => {
                setOpen(expanded ? null : group.id)
                setExample(0)
              }}
              aria-expanded={expanded}
              className="flex w-full items-center gap-3 px-4 py-3 text-left"
            >
              <Dot group={group} />
              <span className="font-medium">{title(group)}</span>
              <span className="font-mono text-xs text-mute">{group.count}번</span>
              <span className="min-w-0 flex-1 truncate text-xs text-mute">{status(group)}</span>
              {group.name && <span className="text-xs font-medium text-brass-deep">확인함</span>}
            </button>

            {expanded && current && (
              <div className="grid gap-4 border-t border-line px-4 pt-3 pb-4">
                <ClipMonitor
                  key={`${group.id}-${current.time}`}
                  slug={song.slug}
                  hasVideo={song.has_video}
                  hasDrums={song.has_drums}
                  time={current.time}
                  start={current.start}
                  end={current.end}
                />
                <div className="flex flex-wrap gap-1.5" role="group" aria-label="대표 장면">
                  {group.examples.map((e, i) => (
                    <button
                      key={e.time}
                      aria-pressed={i === example}
                      onClick={() => setExample(i)}
                      className={`rounded-full px-2.5 py-1 font-mono text-[11px] tabular-nums ${
                        i === example ? 'bg-ink text-paper' : 'bg-ink/6 text-mute hover:text-ink'
                      }`}
                    >
                      {formatClock(e.time)}
                    </button>
                  ))}
                </div>
                <KitPicker value={group.name} disabled={busy} onChange={(name) => void save(group.id, name)} />
                {group.name && (
                  <button
                    disabled={busy}
                    onClick={() => void save(group.id, null)}
                    className="justify-self-start text-xs text-mute underline decoration-line underline-offset-4 hover:text-ink disabled:opacity-50"
                  >
                    {group.detected ? `검출된 대로(${FAMILY_LABEL[group.detected]}) 되돌리기` : '이름 지우기'}
                  </button>
                )}
              </div>
            )}
          </section>
        )
      })}
    </div>
  )
}

function title(group: SoundGroup): string {
  const number = group.id.split('-').pop()
  return group.detected ? `${FAMILY_LABEL[group.detected]} 소리 ${number}` : `놓친 소리 ${number}`
}

function status(group: SoundGroup): string {
  if (group.name === 'ignore') return '악보에서 뺌'
  if (group.name) return `${INST_LABEL[group.name]}로 지정`
  if (!group.detected) return '검출 모델이 놓친 소리 · 이름을 붙이면 악보에 들어갑니다'
  return `${FAMILY_LABEL[group.detected]}로 적힘`
}

// The colour of what the sound is written as now; a dashed ring for a sound not written at all.
function Dot({ group }: { group: SoundGroup }) {
  const family = group.name && group.name !== 'ignore' ? FAMILY[group.name] : group.name ? null : group.detected
  return family ? (
    <span className="size-2.5 shrink-0 rounded-full" style={{ background: FAMILY_COLOR[family] }} />
  ) : (
    <span className="size-2.5 shrink-0 rounded-full border border-dashed border-mute" />
  )
}
