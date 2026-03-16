import { useCallback, useState } from 'react'
import { backendFetch } from '../lib/backend'
import { useAppSettings } from '../contexts/AppSettingsContext'
import { logger } from '../lib/logger'

export type RetakeMode = 'replace_audio_and_video' | 'replace_video' | 'replace_audio'

export interface RetakeSubmitParams {
  videoPath: string
  startTime: number
  duration: number
  prompt: string
  mode: RetakeMode
}

export interface RetakeResult {
  videoPath: string
  videoUrl: string
}

interface UseRetakeState {
  isRetaking: boolean
  retakeStatus: string
  retakeError: string | null
  result: RetakeResult | null
}

export function useRetake() {
  const { settings: appSettings } = useAppSettings()
  const [state, setState] = useState<UseRetakeState>({
    isRetaking: false,
    retakeStatus: '',
    retakeError: null,
    result: null,
  })

  const submitRetake = useCallback(async (params: RetakeSubmitParams) => {
    if (!params.videoPath) return

    setState({
      isRetaking: true,
      retakeStatus: 'Generating',
      retakeError: null,
      result: null,
    })

    try {
      const useComfyui = appSettings.comfyuiEnabled

      const endpoint = useComfyui ? '/api/comfyui/retake' : '/api/retake'
      const body = useComfyui
        ? {
            video_path: params.videoPath,
            start_time: params.startTime,
            duration: params.duration,
            prompt: params.prompt,
            mode: params.mode,
          }
        : {
            video_path: params.videoPath,
            start_time: params.startTime,
            duration: params.duration,
            prompt: params.prompt,
            mode: params.mode,
          }

      const response = await backendFetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })

      const data = await response.json()

      // ComfyUI returns output_paths, native returns video_path
      const videoPath = useComfyui
        ? (data.output_paths?.[0] as string | undefined)
        : (data.video_path as string | undefined)

      if (response.ok && data.status === 'complete' && videoPath) {
        const pathNormalized = videoPath.replace(/\\/g, '/')
        const videoUrl = pathNormalized.startsWith('/') ? `file://${pathNormalized}` : `file:///${pathNormalized}`

        setState({
          isRetaking: false,
          retakeStatus: 'Retake complete!',
          retakeError: null,
          result: {
            videoPath,
            videoUrl,
          },
        })
        return
      }

      const errorMsg = data.error || 'Unknown error'
      setState({
        isRetaking: false,
        retakeStatus: '',
        retakeError: errorMsg,
        result: null,
      })
      logger.error(`Retake failed: ${errorMsg}`)
    } catch (error) {
      const message = (error as Error).message || 'Unknown error'
      logger.error(`Retake error: ${message}`)
      setState({
        isRetaking: false,
        retakeStatus: '',
        retakeError: message,
        result: null,
      })
    }
  }, [appSettings.comfyuiEnabled])

  const resetRetake = useCallback(() => {
    setState({
      isRetaking: false,
      retakeStatus: '',
      retakeError: null,
      result: null,
    })
  }, [])

  return {
    submitRetake,
    resetRetake,
    isRetaking: state.isRetaking,
    retakeStatus: state.retakeStatus,
    retakeError: state.retakeError,
    retakeResult: state.result,
  }
}
