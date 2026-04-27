import i18n from 'i18next';
import LanguageDetector from 'i18next-browser-languagedetector';
import { initReactI18next } from 'react-i18next';
import en from '../public/locales/en/translation.json';
import vi from '../public/locales/vi/translation.json';

const resources = {
  en: { translation: en },
  vi: { translation: vi },
};

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources,
    fallbackLng: 'en',
    debug: false,
    interpolation: {
      escapeValue: false,
    },
  });

export default i18n;
