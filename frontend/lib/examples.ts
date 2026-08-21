// Real questions taken from the eval set itself (data/eval_set.jsonl), not invented —
// every one is verified to clear the calibrated retrieval-confidence threshold
// (tau_abs = 0.03), so a visitor's first try lands on an answer rather than a refusal.
//
// MS-MARCO-XI shares one query_id across all 14 translations, so each entry below is
// literally the same underlying question in every language; the English `gloss` is
// therefore accurate for every row rather than a per-language paraphrase.

export interface ExampleQuestion {
  gloss: string;
  byLanguage: Record<string, string>;
}

// Figures are read off the built corpus (data/corpus_stats.json), not estimated:
// 140,032 passages, ~10,000 per language across 14 languages, 7,783 labelled eval
// queries of which 1,183 are deliberately held out as unanswerable (spec §11.4's
// abstention set).
export const CORPUS_DESCRIPTION = {
  source: "ai4bharat/MSMARCO-XI",
  what: "MS MARCO — real Bing search queries paired with the web passages that answer them — translated into 14 Indian languages by AI4Bharat.",
  stats: [
    { value: "140,032", label: "passages indexed" },
    { value: "14", label: "languages, ~10k each" },
    { value: "7,783", label: "labelled eval queries" },
  ],
  answers:
    "Factual lookups of the kind people type into a search engine: definitions, distances, quantities, durations, health and general-knowledge facts.",
  refuses:
    "Anything outside those passages — personal questions, advice, opinions, or live information. It declines rather than inventing an answer, and shows the confidence score behind the decision.",
};

export const EXAMPLE_QUESTIONS: ExampleQuestion[] = [
  {
    gloss: "How much liquid is in a gallon?",
    byLanguage: {
      as: "এক গেলনত কিমান আউন্স তৰল পদাৰ্থ থাকে",
      bn: "এক গ্যালনে কত আউন্স তরল থাকে।",
      gu: "એક ગેલનમાં કેટલા ઔંસ પ્રવાહી છે",
      hi: "एक गैलन में कितना तरल होता है",
      kn: "ಒಂದು ಗ್ಯಾಲನ್‌ನಲ್ಲಿ ಎಷ್ಟು ಔನ್ಸ್‌ಗಳಷ್ಟು ದ್ರವವಿದೆ",
      ml: "ഒരു ഗാലനിൽ എത്ര ഔൺസ് ദ്രാവകമുണ്ട്",
      ne: "एउटा ग्यालनमा कति औंस तरल पदार्थ हुन्छ।",
      or: "ଗ୍ୟାଲନ ପିଛା କେତେ ଆଉନ୍ସ ତରଳ ପଦାର୍ଥ ଅଛି",
      pa: "ਇੱਕ ਗੈਲਨ ਵਿੱਚ ਕਿੰਨਾ ਔਂਸ ਤਰਲ ਹੁੰਦਾ ਹੈ",
      ta: "ஒரு கேலனில் எவ்வளவு அவுன்ஸ் திரவம் உள்ளது",
      te: "ఒక గ్యాలన్‌లో ఎన్ని ఔన్సుల ద్రవం ఉంటుంది",
      ur: "ایک گیلن میں کتنے اونس مائع ہوتے ہیں",
    },
  },
  {
    gloss: "Distance from Scottsdale to the Grand Canyon",
    byLanguage: {
      as: "দূৰত্ব স্কটছডেলৰ পৰা গ্ৰেণ্ড কেনিয়নলৈ",
      bn: "দূরত্ব স্কটসডেল থেকে গ্র্যান্ড ক্যানিয়ন",
      gu: "ડિસ્ટન્સ સ્કોટ્સડેલ ટુ ગ્રાન્ડ કેન્યોન",
      hi: "दूरी स्कॉट्सडेल से ग्रैंड कैन्यन तक",
      kn: "ದೂರದ ಸ್ಕಾಟ್ಸ್‌ಡೇಲ್‌ನಿಂದ ಗ್ರ್ಯಾಂಡ್ ಕ್ಯಾನ್ಯನ್‌ವರೆಗೆ",
      ml: "സ്‌കോട്‌സ്‌ഡെയ്‌ലിനും ഗ്രാൻഡ് കാന്യനും ഇടയിലുള്ള ദൂരം",
      mr: "दूरचे स्कॉट्सडेल ते ग्रँड कॅन्यन",
      ne: "दुरी स्कट्सडेलदेखि ग्रान्ड क्यान्यनसम्म",
      or: "ଦୂରତା ସ୍କଟ୍ସଡେଲରୁ ଗ୍ରାଣ୍ଡ କ୍ୟାନିୟନ ପର୍ଯ୍ୟନ୍ତ",
      pa: "ਡਿਸਟੈਂਸ ਸਕਾਟਸਡੇਲ ਤੋਂ ਗ੍ਰੈਂਡ ਕੈਨਿਯਨ ਤੱਕ",
      sa: "दूरं स्काट्स्डेल् इत्यतः ग्राण्ड्-केन्यन् पर्यन्तम्।",
      ta: "தூர ஸ்காட்ஸ்டேல் முதல் கிராண்ட் கேன்யன் வரை",
      te: "దూరం స్కాట్స్‌డేల్ నుండి గ్రాండ్ క్యాన్యాన్ వరకు",
      ur: "فاصلہ اسکاٹسڈیل سے گرینڈ کینین تک",
    },
  },
  {
    gloss: "Definition of honesty or integrity",
    byLanguage: {
      as: "সততা বা সততার সংজ্ঞা",
      bn: "সততা বা সততার সংজ্ঞা",
      gu: "પ્રામાણિકતા અથવા સત્યનિષ્ઠાની વ્યાખ્યા",
      hi: "ईमानदारी या सच्चाई की परिभाषा",
      kn: "ಪ್ರಾಮಾಣಿಕತೆ ಅಥವಾ ಸಮಗ್ರತೆಯ ವ್ಯಾಖ್ಯಾನ",
      ml: "സത്യസന്ധതയോ സത്യസന്ധതയുടെ നിർവചനമോ",
      mr: "प्रामाणिकपणा किंवा सचोटीची व्याख्या",
      or: "ସତ୍ୟତା ବା ସତ୍ୟନିଷ୍ଠତା ସଂଜ୍ଞା",
      pa: "ਇਮਾਨਦਾਰੀ ਜਾਂ ਇਮਾਨਦਾਰੀ ਦੀ ਪਰਿਭਾਸ਼ਾ",
      sa: "नैष्ठिकी वा सत्यनिष्ठा वा परिभाषा",
      ta: "நேர்மை அல்லது நேர்மையின் வரையறை",
      te: "నిజాయితీ లేదా సమగ్రత నిర్వచనం",
      ur: "دیانت داری یا دیانت کی تعریف",
    },
  },
  {
    gloss: "Do you need a degree to be a gym teacher?",
    byLanguage: {
      as: "জিম শিক্ষক হ'বলৈ আপোনাক ডিগ্ৰী লাগিব নেকি",
      bn: "জিমের শিক্ষক হতে আপনার কি ডিগ্রি প্রয়োজন?",
      gu: "શું તમને જિમ શિક્ષક બનવા માટે ડિગ્રીની જરૂર છે?",
      hi: "क्या आपको जिम शिक्षक बनने के लिए डिग्री की आवश्यकता है?",
      kn: "ಜಿಮ್ ಶಿಕ್ಷಕರಾಗಲು ನಿಮಗೆ ಪದವಿ ಬೇಕೇ?",
      ml: "ജിം അദ്ധ്യാപകനാകാൻ നിങ്ങൾക്ക് ഒരു ബിരുദം ആവശ്യമുണ്ടോ",
      mr: "व्यायामशाळेचे शिक्षक होण्यासाठी तुम्हाला पदवीधर होणे आवश्यक आहे का?",
      ne: "के तपाईँलाई जिम शिक्षक बन्नाका लागि डिग्री चाहिन्छ?",
      or: "ଜିମ୍ ଶିକ୍ଷକ ହେବା ପାଇଁ ଆପଣଙ୍କୁ ଡିଗ୍ରୀ ଦରକାର କି?",
      pa: "ਕੀ ਤੁਹਾਨੂੰ ਜਿਮ ਅਧਿਆਪਕ ਬਣਨ ਲਈ ਡਿਗਰੀ ਦੀ ਲੋੜ ਹੈ?",
      sa: "जिम्-शिक्षकः भवितुं भवतः किमपि उपाधिका अपेक्षिता अस्ति वा?",
      ta: "ஜிம்மில் ஆசிரியராக வேண்டுமானால் உங்களுக்கு பட்டம் தேவையா?",
      ur: "کیا آپ کو جم ٹیچر بننے کے لیے ڈگری کی ضرورت ہے",
    },
  },
  {
    gloss: "Does a conviction mean jail?",
    byLanguage: {
      as: "দোষী সাব্যস্ত হোৱাৰ অৰ্থ কি কাৰাগাৰ?",
      bn: "দোষী সাব্যস্ত হওয়ার অর্থ কি কারাগার?",
      gu: "શું દોષિત ઠેરવવું એટલે જેલ?",
      hi: "क्या दोषसिद्धि का मतलब जेल है?",
      kn: "ತೀರ್ಪು ಎಂದರೆ ಜೈಲು ಎಂದರೇನು?",
      ml: "ശിക്ഷാവിധി എന്നാൽ ജയിൽ എന്നാണോ?",
      ne: "के दोषसिद्धिको अर्थ जेल हो?",
      or: "ଦୋଷୀ ସାବ୍ୟସ୍ତ ହେବା ମାନେ କି ଜେଲ?",
      pa: "ਕੀ ਸਜ਼ਾ ਦਾ ਮਤਲਬ ਹੈ ਜੇਲ੍ਹ?",
      sa: "दोषसिद्धिः कारागारः इति किं?",
      ta: "தண்டனை என்பது சிறை என்றால் என்ன?",
      te: "దోషిగా నిర్ధారించబడినందుకు జైలు అని అర్థంగా ఉంటుందా?",
      ur: "کیا سزا کا مطلب ہے جیل؟",
    },
  },
];

// Questions the corpus genuinely cannot answer — shown alongside the working ones so
// the refusal path reads as designed behavior rather than breakage. Spec §18 wants the
// system demonstrably refusing off-topic questions; naming them up front makes that
// demonstrable rather than accidental.
export const OUT_OF_SCOPE_EXAMPLES = [
  "What's your name?",
  "It's hot here — what should I do?",
  "What language am I speaking?",
];

const FALLBACK_LANGUAGE = "hi";

// Deliberately script-diverse: Devanagari, Tamil, Bengali, Telugu, Perso-Arabic (RTL),
// Kannada, Malayalam, Gurmukhi. Under "Auto-detect" the examples rotate through these
// so the very first thing a visitor sees demonstrates the multilingual claim, instead
// of five Hindi rows implying the system is Hindi-only — which is exactly how the
// earlier Hindi-fallback version read.
const AUTO_SHOWCASE_LANGUAGES = ["hi", "ta", "bn", "te", "ur", "kn", "ml", "pa"];

/** Sarvam codes are BCP-47 ("hi-IN"); the corpus keys are bare ISO codes ("hi").
 *  Odia is the one mismatch — Sarvam says "od-IN", the corpus says "or". */
export function corpusLanguageKey(sarvamCode: string): string {
  if (sarvamCode === "auto") return FALLBACK_LANGUAGE;
  const bare = sarvamCode.split("-")[0];
  return bare === "od" ? "or" : bare;
}

export interface ResolvedExample {
  gloss: string;
  text: string;
  languageCode: string;
}

export function examplesForLanguage(sarvamCode: string): ResolvedExample[] {
  if (sarvamCode !== "auto") {
    const key = corpusLanguageKey(sarvamCode);
    return EXAMPLE_QUESTIONS.map((e) => ({
      gloss: e.gloss,
      text: e.byLanguage[key] ?? "",
      languageCode: key,
    })).filter((e) => Boolean(e.text));
  }

  // Auto-detect: give each example a different language, preferring an unused one from
  // the showcase list so the set stays script-diverse even when a particular question
  // is missing a translation (not every query_id covers all 14).
  const used = new Set<string>();
  return EXAMPLE_QUESTIONS.map((e) => {
    const available = AUTO_SHOWCASE_LANGUAGES.filter((l) => e.byLanguage[l]);
    const pick =
      available.find((l) => !used.has(l)) ?? available[0] ?? Object.keys(e.byLanguage)[0];
    if (pick) used.add(pick);
    return {
      gloss: e.gloss,
      text: pick ? e.byLanguage[pick] : e.byLanguage[FALLBACK_LANGUAGE],
      languageCode: pick ?? FALLBACK_LANGUAGE,
    };
  }).filter((e) => Boolean(e.text));
}
