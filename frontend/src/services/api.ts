import axios from 'axios'
import type { DiagnosisResult, DiagnosisListItem, ModelsStatusResponse, PredictResult } from '../types'

const api = axios.create({ baseURL: '/api' })

export async function runDiagnosis(
  file: File,
  modelName: string,
  generateGradcam: boolean,
): Promise<DiagnosisResult> {
  const form = new FormData()
  form.append('file', file)
  form.append('model_name', modelName)
  form.append('generate_gradcam_flag', String(generateGradcam))
  const { data } = await api.post<DiagnosisResult>('/diagnosis/', form)
  return data
}

export async function runPredict(file: File): Promise<PredictResult> {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post<PredictResult>('/predict/', form)
  return data
}

export async function getHistory(limit = 50, offset = 0): Promise<DiagnosisListItem[]> {
  const { data } = await api.get<DiagnosisListItem[]>('/history/', { params: { limit, offset } })
  return data
}

export async function getDiagnosis(id: number): Promise<DiagnosisResult> {
  const { data } = await api.get<DiagnosisResult>(`/history/${id}`)
  return data
}

export async function deleteDiagnosis(id: number): Promise<void> {
  await api.delete(`/history/${id}`)
}

export async function getModelsStatus(): Promise<ModelsStatusResponse> {
  const { data } = await api.get<ModelsStatusResponse>('/models/')
  return data
}
