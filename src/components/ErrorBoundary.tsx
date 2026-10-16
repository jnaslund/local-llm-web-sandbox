import { Component, type ErrorInfo, type ReactNode } from 'react'

type Props = { name: string; children: ReactNode }
type State = { error: Error | null }

/** Keeps a crash in one panel from unmounting the whole app. */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(`[ui] ${this.props.name} crashed:`, error, info.componentStack)
  }

  render() {
    if (!this.state.error) return this.props.children
    return (
      <div className="space-y-2 rounded-lg bg-red-50 p-3 text-xs text-red-800 dark:bg-red-950 dark:text-red-300">
        <p className="font-medium">The {this.props.name} panel crashed.</p>
        <p className="font-mono break-words">{this.state.error.message}</p>
        <button className="btn-secondary" onClick={() => this.setState({ error: null })}>
          Try again
        </button>
      </div>
    )
  }
}
