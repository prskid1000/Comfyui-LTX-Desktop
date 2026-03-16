import { useCallback, useState } from 'react'
import { backendFetch } from '../lib/backend'
import { useAppSettings } from '../contexts/AppSettingsContext'
import { logger } from '../lib/logger'

export type IcLoraConditioningType = 'canny' | 'depth' | 'pose'

export interface IcLoraSubmitParams {
  videoPath: string
  conditioningType: IcLoraConditioningType
  conditioningStrength: number
  prompt: string
}

export interface IcLoraResult {
  videoPath: string
  videoUrl: string
}

interface UseIcLoraState {
  isGenerating: boolean
  status: string
  error: string | null
  result: IcLoraResult | null
}

export function useIcLora() {
  const { settings: appSettings } = useAppSettings()
  const [state, setState] = useState<UseIcLoraState>({
    isGenerating: false,
    status: '',
    error: null,
    result: null,
  })

  const submitIcLora = useCallback(async (params: IcLoraSubmitParams) => {
    if (!params.videoPath || !params.prompt.trim()) return

    setState({
      isGenerating: true,
      status: 'Generating',
      error: null,
      result: null,
    })

    try {
      const useComfyui = appSettings.comfyuiEnabled

      const endpoint = useComfyui ? '/api/comfyui/ic-lora/generate' : '/api/ic-lora/generate'
      const body = useComfyui
        ? {
            video_path: params.videoPath,
            conditioning_type: params.conditioningType,
            conditioning_strength: params.conditioningStrength,
            prompt: params.prompt,
          }
        : {
            video_path: params.videoPath,
            conditioning_type: params.conditioningType,
            conditioning_strength: params.conditioningStrength,
            prompt: params.prompt,
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
          isGenerating: false,
          status: 'Generation complete!',
          error: null,
          result: {
            videoPath,
            videoUrl,
          },
        })
        return
      }

      const errorMsg = data.error || 'Unknown error'
      logger.error(`IC-LoRA failed: ${errorMsg}`)
      setState({
        isGenerating: false,
        status: '',
        error: errorMsg,
        result: null,
      })
    } catch (error) {
      const message = (error as Error).message || 'Unknown error'
      logger.error(`IC-LoRA error: ${message}`)
      setState({
        isGenerating: false,
        status: '',
        error: message,
        result: null,
      })
    }
  }, [appSettings.comfyuiEnabled])

  const reset = useCallback(() => {
    setState({
      isGenerating: false,
      status: '',
      error: null,
      result: null,
    })
  }, [])

  return {
    submitIcLora,
    resetIcLora: reset,
    isIcLoraGenerating: state.isGenerating,
    icLoraStatus: state.status,
    icLoraError: state.error,
    icLoraResult: state.result,
  }
}
