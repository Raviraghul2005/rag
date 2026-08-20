// Mirrors backend/app/stt/language_codes.py's CORPUS_TO_SARVAM table. The corpus's 14
// languages, each with its Sarvam BCP-47 code for the STT WebSocket's `language_code`
// param. Spec §9 asked for the STT language-coverage gap to be stated explicitly; as
// of this build there is no gap — Sarvam's saaras:v3-realtime covers all 14 (see the
// backend module's comment for the source and date verified).
export interface LanguageOption {
  code: string; // corpus 2-letter code
  sarvamCode: string; // Sarvam BCP-47 code, passed to /ws/transcribe
  label: string;
  nativeLabel: string;
}

export const LANGUAGES: LanguageOption[] = [
  { code: "hi", sarvamCode: "hi-IN", label: "Hindi", nativeLabel: "हिन्दी" },
  { code: "bn", sarvamCode: "bn-IN", label: "Bengali", nativeLabel: "বাংলা" },
  { code: "ta", sarvamCode: "ta-IN", label: "Tamil", nativeLabel: "தமிழ்" },
  { code: "te", sarvamCode: "te-IN", label: "Telugu", nativeLabel: "తెలుగు" },
  { code: "kn", sarvamCode: "kn-IN", label: "Kannada", nativeLabel: "ಕನ್ನಡ" },
  { code: "ml", sarvamCode: "ml-IN", label: "Malayalam", nativeLabel: "മലയാളം" },
  { code: "mr", sarvamCode: "mr-IN", label: "Marathi", nativeLabel: "मराठी" },
  { code: "gu", sarvamCode: "gu-IN", label: "Gujarati", nativeLabel: "ગુજરાતી" },
  { code: "pa", sarvamCode: "pa-IN", label: "Punjabi", nativeLabel: "ਪੰਜਾਬੀ" },
  { code: "or", sarvamCode: "od-IN", label: "Odia", nativeLabel: "ଓଡ଼ିଆ" },
  { code: "ur", sarvamCode: "ur-IN", label: "Urdu", nativeLabel: "اردو" },
  { code: "ne", sarvamCode: "ne-IN", label: "Nepali", nativeLabel: "नेपाली" },
  { code: "as", sarvamCode: "as-IN", label: "Assamese", nativeLabel: "অসমীয়া" },
  { code: "sa", sarvamCode: "sa-IN", label: "Sanskrit", nativeLabel: "संस्कृतम्" },
];

export const AUTO_DETECT: LanguageOption = {
  code: "auto",
  sarvamCode: "auto",
  label: "Auto-detect",
  nativeLabel: "Auto-detect",
};
