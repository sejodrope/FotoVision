import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, ImageIcon, X } from 'lucide-react'
import clsx from 'clsx'

interface Props {
  onFileSelected: (file: File, preview: string) => void
  disabled?: boolean
}

export function ImageUploader({ onFileSelected, disabled }: Props) {
  const [preview, setPreview] = useState<string | null>(null)
  const [fileName, setFileName] = useState<string | null>(null)

  const onDrop = useCallback(
    (accepted: File[]) => {
      if (!accepted.length) return
      const file = accepted[0]
      const url = URL.createObjectURL(file)
      setPreview(url)
      setFileName(file.name)
      onFileSelected(file, url)
    },
    [onFileSelected],
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpg', '.jpeg', '.png', '.webp'] },
    multiple: false,
    disabled,
  })

  const clear = (e: React.MouseEvent) => {
    e.stopPropagation()
    setPreview(null)
    setFileName(null)
  }

  return (
    <div
      {...getRootProps()}
      className={clsx(
        'relative border-2 border-dashed rounded-2xl flex flex-col items-center justify-center cursor-pointer transition-all min-h-[280px]',
        isDragActive ? 'border-primary-500 bg-primary-50' : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50',
        disabled && 'opacity-50 cursor-not-allowed',
      )}
    >
      <input {...getInputProps()} />
      {preview ? (
        <div className="relative w-full h-64">
          <img src={preview} alt="preview" className="w-full h-full object-contain rounded-xl" />
          <button
            onClick={clear}
            className="absolute top-2 right-2 bg-white rounded-full p-1 shadow hover:bg-red-50"
          >
            <X className="w-4 h-4 text-red-500" />
          </button>
          <p className="absolute bottom-2 left-0 right-0 text-center text-xs text-gray-500 bg-white/80 py-1">
            {fileName}
          </p>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-3 p-8 text-center">
          {isDragActive ? (
            <ImageIcon className="w-12 h-12 text-primary-500" />
          ) : (
            <Upload className="w-12 h-12 text-gray-400" />
          )}
          <div>
            <p className="text-base font-medium text-gray-700">
              {isDragActive ? 'Solte a imagem aqui' : 'Arraste uma foto da folha'}
            </p>
            <p className="text-sm text-gray-500 mt-1">ou clique para selecionar</p>
          </div>
          <p className="text-xs text-gray-400">JPG, PNG, WEBP — máx. 10MB</p>
        </div>
      )}
    </div>
  )
}
