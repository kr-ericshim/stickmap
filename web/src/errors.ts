// Error text for the banner, in the user's words rather than the browser's.

export const OFFLINE_MESSAGE = '엔진에 연결할 수 없습니다. 다시 연결하는 중…'

const NETWORK = /failed to fetch|networkerror|network request failed|load failed|err_connection/i

/** True when the engine could not be reached at all (no HTTP answer). */
export function isConnectionError(cause: unknown): boolean {
  if (cause instanceof TypeError) return NETWORK.test(cause.message)
  if (typeof cause === 'string') return NETWORK.test(cause)
  if (cause instanceof Error) return NETWORK.test(cause.message)
  return false
}

/**
 * A message fit for the error banner. Connection failures become one plain sentence;
 * server messages (e.g. a 409 `detail`) are already written for the user and pass through.
 */
export function friendlyError(cause: unknown): string {
  if (isConnectionError(cause)) return OFFLINE_MESSAGE
  if (cause instanceof Error) return cause.message || '알 수 없는 오류가 났습니다.'
  if (typeof cause === 'string') return cause || '알 수 없는 오류가 났습니다.'
  return String(cause)
}

/** The user-facing message of a failed response: its JSON `detail` when present, else the status. */
export async function responseError(response: Response): Promise<string> {
  const body = await response.json().catch(() => null)
  const detail = body?.detail
  if (typeof detail === 'string' && detail) return detail
  return `${response.status} ${response.statusText}`.trim()
}
