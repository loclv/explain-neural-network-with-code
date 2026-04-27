/**
 * WhyMatrix — child-friendly explanation of why neural networks need matrices.
 *
 * Uses an analogy of friends passing notes in class, with a small
 * interactive grid showing weights as connection strengths.
 */

import { useState } from 'react';
import { useTranslation } from 'react-i18next';

export default function WhyMatrix() {
  const { t } = useTranslation();
  const [hovered, setHovered] = useState<number | null>(null);

  const friends = t('whyMatrix.friends', { returnObjects: true }) as string[];
  const strengths = [0.8, 0.3, 1.0, 0.1, 0.9, 0.5];

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-5 mb-8">
      <h2 className="text-lg font-semibold mb-3">{t('whyMatrix.title')}</h2>

      <div className="space-y-4 text-sm text-slate-700 leading-relaxed">
        <p>{t('whyMatrix.p1')}</p>
        <p>{t('whyMatrix.p2')}</p>

        <div className="bg-indigo-50 border border-indigo-100 rounded-lg p-4">
          <p className="font-medium text-indigo-900 mb-3">
            {t('whyMatrix.gridTitle')}
          </p>
          <div className="grid grid-cols-3 gap-2 max-w-xs">
            {friends.map((name, i) => (
              <button
                type="button"
                key={name}
                className={`rounded-lg px-3 py-2 text-xs font-medium transition-all ${
                  hovered === i
                    ? 'bg-indigo-600 text-white scale-105 shadow-md'
                    : 'bg-white text-slate-600 border border-slate-200 hover:border-indigo-300'
                }`}
                onMouseEnter={() => setHovered(i)}
                onMouseLeave={() => setHovered(null)}
              >
                {name}
                <span
                  className={`ml-1 inline-block w-2 h-2 rounded-full ${
                    strengths[i] > 0.7
                      ? 'bg-green-400'
                      : strengths[i] > 0.4
                        ? 'bg-amber-400'
                        : 'bg-red-400'
                  }`}
                />
              </button>
            ))}
          </div>
          {hovered !== null && (
            <p className="mt-2 text-xs text-indigo-700">
              {t('whyMatrix.strength', {
                name: friends[hovered],
                value: strengths[hovered],
              })}{' '}
              {strengths[hovered] > 0.7
                ? t('whyMatrix.strong')
                : strengths[hovered] > 0.4
                  ? t('whyMatrix.medium')
                  : t('whyMatrix.weak')}
            </p>
          )}
        </div>

        <p>{t('whyMatrix.p3')}</p>

        <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
          <p className="font-medium text-amber-900 mb-1">
            {t('whyMatrix.analogyTitle')}
          </p>
          <ul className="list-disc pl-4 space-y-1 text-amber-900">
            <li>{t('whyMatrix.analogy1')}</li>
            <li>{t('whyMatrix.analogy2')}</li>
            <li>{t('whyMatrix.analogy3')}</li>
          </ul>
        </div>

        <p>{t('whyMatrix.p4')}</p>
      </div>
    </div>
  );
}
