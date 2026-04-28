export type ModelId = 'mobilenet_v2' | 'resnet50' | 'efficientnet_b0' | 'vit_b_16'

export type ClassId =
  | 'saudavel'
  | 'mildio'
  | 'oidio'
  | 'clorose_nitrogenio'
  | 'danos_pragas'
  | 'estresse_hidrico'

export interface DiagnosisResult {
  id: number
  created_at: string
  model_name: ModelId
  predicted_class: ClassId
  predicted_label: string
  confidence: number
  scores: Record<ClassId, number>
  gradcam_image: string | null
  demo_mode: boolean
  calibrated: boolean
  original_filename: string | null
}

export interface DiagnosisListItem {
  id: number
  created_at: string
  model_name: ModelId
  predicted_class: ClassId
  predicted_label: string
  confidence: number
  demo_mode: boolean
  original_filename: string | null
}

export interface ModelInfo {
  id: ModelId
  label: string
  calibrated: boolean
  description: string
}

export interface ModelsStatusResponse {
  models: ModelInfo[]
  demo_mode: boolean
}

export const CLASS_LABELS: Record<ClassId, string> = {
  saudavel: 'Saudável',
  mildio: 'Míldio',
  oidio: 'Oídio',
  clorose_nitrogenio: 'Clorose por Nitrogênio',
  danos_pragas: 'Danos por Pragas',
  estresse_hidrico: 'Estresse Hídrico',
}

export const CLASS_LABELS_SHORT: Record<ClassId, string> = {
  saudavel: 'Saudável',
  mildio: 'Míldio',
  oidio: 'Oídio',
  clorose_nitrogenio: 'Clorose N',
  danos_pragas: 'Pragas',
  estresse_hidrico: 'Est. Hídrico',
}

export const CLASS_COLORS: Record<ClassId, string> = {
  saudavel: '#22c55e',
  mildio: '#ef4444',
  oidio: '#f97316',
  clorose_nitrogenio: '#eab308',
  danos_pragas: '#8b5cf6',
  estresse_hidrico: '#3b82f6',
}

export const MODEL_LABELS: Record<ModelId, string> = {
  mobilenet_v2: 'MobileNetV2',
  resnet50: 'ResNet-50',
  efficientnet_b0: 'EfficientNet-B0',
  vit_b_16: 'ViT-B/16',
}
