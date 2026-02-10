interface ToastItem {
  id: number;
  message: string;
  type: 'success' | 'error' | 'info';
}

const bgMap = {
  success: 'bg-green-600',
  error: 'bg-red-600',
  info: 'bg-indigo-600',
};

export default function Toast({ toasts, onRemove }: { toasts: ToastItem[]; onRemove: (id: number) => void }) {
  if (!toasts.length) return null;

  return (
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-2">
      {toasts.map((t) => (
        <div
          key={t.id}
          className={`${bgMap[t.type]} text-white px-4 py-2.5 rounded-lg shadow-lg text-sm flex items-center gap-3 animate-[slideIn_0.2s_ease]`}
        >
          <span>{t.message}</span>
          <button onClick={() => onRemove(t.id)} className="text-white/70 hover:text-white ml-2">
            x
          </button>
        </div>
      ))}
    </div>
  );
}
