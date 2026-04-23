import { create } from 'zustand'
import type { DiagnosisResult, ModelId } from '../types'

interface DiagnosisStore {
  lastResult: DiagnosisResult | null
  selectedModel: ModelId
  generateGradcam: boolean
  setLastResult: (result: DiagnosisResult) => void
  setSelectedModel: (model: ModelId) => void
  setGenerateGradcam: (v: boolean) => void
  clearResult: () => void
}

export const useDiagnosisStore = create<DiagnosisStore>((set) => ({
  lastResult: null,
  selectedModel: 'mobilenet_v2',
  generateGradcam: true,
  setLastResult: (result) => set({ lastResult: result }),
  setSelectedModel: (model) => set({ selectedModel: model }),
  setGenerateGradcam: (v) => set({ generateGradcam: v }),
  clearResult: () => set({ lastResult: null }),
}))
