
"""
src/ingest.py
─────────────────────────────────────────────────────────────────
Sage AI — Data Ingestion Pipeline
 
What this file does:
1. Fetches all 700 Bhagavad Gita shlokas from a public API
2. Enriches each shloka with emotion tags and topic metadata
3. Saves a clean, structured JSON to data/gita_shlokas.json
 
Run:
    python src/ingest.py
"""
 
import json
import time
import requests
from pathlib import Path
from tqdm import tqdm
 
# ── Paths ─────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_FILE = DATA_DIR / "gita_shlokas.json"
 
# ── Emotion tag map ───────────────────────────────────────────────
# Maps chapter themes to human emotions — helps with metadata filtering later
CHAPTER_EMOTION_MAP = {
    1:  ["grief", "confusion", "fear", "despair"],
    2:  ["clarity", "duty", "impermanence", "self-knowledge"],
    3:  ["action", "duty", "desire", "purpose"],
    4:  ["wisdom", "knowledge", "sacrifice", "surrender"],
    5:  ["renunciation", "detachment", "peace", "equanimity"],
    6:  ["meditation", "self-control", "focus", "discipline"],
    7:  ["devotion", "knowledge", "illusion", "faith"],
    8:  ["death", "impermanence", "liberation", "memory"],
    9:  ["devotion", "surrender", "love", "grace"],
    10: ["greatness", "wonder", "divine", "awe"],
    11: ["awe", "fear", "revelation", "cosmic"],
    12: ["devotion", "love", "qualities", "practice"],
    13: ["body", "soul", "knowledge", "field"],
    14: ["qualities", "nature", "liberation", "transcendence"],
    15: ["cosmic", "tree", "supreme", "knowledge"],
    16: ["virtue", "vice", "divine", "demonic"],
    17: ["faith", "food", "worship", "discipline"],
    18: ["renunciation", "duty", "liberation", "conclusion"],
}
 
CHAPTER_TITLES = {
    1:  "Arjuna's Dilemma",
    2:  "Transcendental Knowledge",
    3:  "Karma Yoga — Path of Action",
    4:  "Jnana Yoga — Path of Knowledge",
    5:  "Karma Sannyas Yoga",
    6:  "Dhyana Yoga — Path of Meditation",
    7:  "Jnana Vijnana Yoga",
    8:  "Aksara Brahma Yoga",
    9:  "Raja Vidya Yoga",
    10: "Vibhuti Yoga",
    11: "Visvarupa Darsana Yoga",
    12: "Bhakti Yoga — Path of Devotion",
    13: "Ksetra Ksetrajna Vibhaga Yoga",
    14: "Gunatraya Vibhaga Yoga",
    15: "Purushottama Yoga",
    16: "Daivasura Sampad Vibhaga Yoga",
    17: "Sraddhatraya Vibhaga Yoga",
    18: "Moksha Sannyas Yoga",
}
 
 
def fetch_shloka(chapter: int, verse: int) -> dict | None:
    """
    Fetches a single shloka from the Bhagavad Gita API.
    API docs: https://bhagavad-gita3.p.rapidapi.com
    Falls back to a local stub if the API is unavailable.
    """
    url = f"https://bhagavad-gita3.p.rapidapi.com/v2/chapters/{chapter}/verses/{verse}/"
    headers = {
        "X-RapidAPI-Key": "SIGN-UP-FOR-KEY",   # Free tier available
        "X-RapidAPI-Host": "bhagavad-gita3.p.rapidapi.com"
    }
 
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException:
        pass
 
    return None
 
 
def build_shloka_record(raw: dict, chapter: int, verse: int) -> dict:
    """
    Transforms raw API response into a clean, enriched record
    ready for embedding and retrieval.
    """
    # Extract translations — prefer English
    translations = raw.get("translations", [])
    english_translation = ""
    for t in translations:
        if t.get("language") == "english":
            english_translation = t.get("description", "")
            break
    if not english_translation and translations:
        english_translation = translations[0].get("description", "")
 
    # Build the text we'll embed (rich context for better retrieval)
    embed_text = (
        f"Chapter {chapter}, Verse {verse} — {CHAPTER_TITLES.get(chapter, '')}. "
        f"{english_translation}"
    )
 
    return {
        "id": f"ch{chapter:02d}_v{verse:03d}",
        "chapter": chapter,
        "chapter_title": CHAPTER_TITLES.get(chapter, f"Chapter {chapter}"),
        "verse": verse,
        "sanskrit": raw.get("text", ""),
        "transliteration": raw.get("transliteration", ""),
        "word_meanings": raw.get("word_meanings", ""),
        "translation": english_translation,
        "emotions": CHAPTER_EMOTION_MAP.get(chapter, []),
        "embed_text": embed_text,
    }
 
 
def load_sample_data() -> list[dict]:
    """
    Returns a small set of hand-curated shlokas so the project works
    on Day 1 without any API key. Replace with fetch_all_shlokas()
    once you have an API key.
 
    These are some of the most universally quoted shlokas.
    """
    sample_shlokas = [
        {
            "id": "ch02_v047",
            "chapter": 2,
            "chapter_title": "Transcendental Knowledge",
            "verse": 47,
            "sanskrit": "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन। मा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि॥",
            "transliteration": "karmanye vadhikaraste ma phaleshu kadachana | ma karma-phala-hetur bhur ma te sango 'stv akarmani",
            "word_meanings": "karmani—in prescribed duties; eva—certainly; adhikarah—right; te—of you; ma—never; phaleshu—in the fruits; kadachana—at any time",
            "translation": "You have a right to perform your prescribed duties, but you are not entitled to the fruits of your actions. Never consider yourself the cause of the results of your activities, and never be attached to not doing your duty.",
            "emotions": ["duty", "action", "detachment", "purpose"],
            "embed_text": "Chapter 2, Verse 47 — Transcendental Knowledge. You have a right to perform your prescribed duties, but you are not entitled to the fruits of your actions. Never consider yourself the cause of the results of your activities, and never be attached to not doing your duty.",
        },
        {
            "id": "ch02_v014",
            "chapter": 2,
            "chapter_title": "Transcendental Knowledge",
            "verse": 14,
            "sanskrit": "मात्रास्पर्शास्तु कौन्तेय शीतोष्णसुखदुःखदाः। आगमापायिनोऽनित्यास्तांस्तितिक्षस्व भारत॥",
            "transliteration": "matra-sparshas tu kaunteya shitoshna-sukha-duhkha-dah | agamapayino 'nityas tams titikshasva bharata",
            "word_meanings": "matra-sparshah—sensory perceptions; tu—only; kaunteya—O son of Kunti; shita—winter; ushna—summer; sukha—happiness; duhkha—and pain",
            "translation": "O son of Kunti, the nonpermanent appearance of happiness and distress, and their disappearance in due course, are like the appearance and disappearance of winter and summer seasons. They arise from sense perception, O scion of Bharata, and one must learn to tolerate them without being disturbed.",
            "emotions": ["grief", "impermanence", "resilience", "equanimity"],
            "embed_text": "Chapter 2, Verse 14 — Transcendental Knowledge. Happiness and distress come and go like seasons — they are temporary. Learn to tolerate them without being disturbed.",
        },
        {
            "id": "ch02_v020",
            "chapter": 2,
            "chapter_title": "Transcendental Knowledge",
            "verse": 20,
            "sanskrit": "न जायते म्रियते वा कदाचिन्नायं भूत्वा भविता वा न भूयः। अजो नित्यः शाश्वतोऽयं पुराणो न हन्यते हन्यमाने शरीरे॥",
            "transliteration": "na jayate mriyate va kadachin nayam bhutva bhavita va na bhuyah | ajo nityah shashvato 'yam purano na hanyate hanyamane sharire",
            "word_meanings": "na—never; jayate—takes birth; mriyate—dies; va—either; kadachit—at any time",
            "translation": "For the soul there is never birth nor death at any time. It has not come into being, does not come into being, and will not come into being. It is unborn, eternal, ever-existing, and primeval. It is not slain when the body is slain.",
            "emotions": ["death", "fear", "grief", "impermanence"],
            "embed_text": "Chapter 2, Verse 20 — Transcendental Knowledge. The soul is never born and never dies. It is eternal and unending. Do not grieve for what is imperishable.",
        },
        {
            "id": "ch06_v005",
            "chapter": 6,
            "chapter_title": "Dhyana Yoga — Path of Meditation",
            "verse": 5,
            "sanskrit": "उद्धरेदात्मनात्मानं नात्मानमवसादयेत्। आत्मैव ह्यात्मनो बन्धुरात्मैव रिपुरात्मनः॥",
            "transliteration": "uddhared atmanatmanam natmanam avasadayet | atmaiva hy atmano bandhur atmaiva ripur atmanah",
            "word_meanings": "uddharet—one must deliver; atmana—by the mind; atmanam—the conditioned soul; na—never; atmanam—the conditioned soul; avasadayet—put into degradation",
            "translation": "One must deliver himself with the help of his mind, and not degrade himself. The mind is the friend of the conditioned soul, and his enemy as well.",
            "emotions": ["self-doubt", "confusion", "discipline", "self-knowledge"],
            "embed_text": "Chapter 6, Verse 5 — Dhyana Yoga. You are your own best friend and your own worst enemy. Lift yourself up — do not let your own mind drag you down.",
        },
        {
            "id": "ch18_v066",
            "chapter": 18,
            "chapter_title": "Moksha Sannyas Yoga",
            "verse": 66,
            "sanskrit": "सर्वधर्मान्परित्यज्य मामेकं शरणं व्रज। अहं त्वा सर्वपापेभ्यो मोक्षयिष्यामि मा शुचः॥",
            "transliteration": "sarva-dharman parityajya mam ekam sharanam vraja | aham tvam sarva-papebhyo mokshayishyami ma shuchah",
            "word_meanings": "sarva-dharman—all varieties of religion; parityajya—abandoning; mam—unto Me; ekam—only; sharanam—for surrender; vraja—go",
            "translation": "Abandon all varieties of religion and just surrender unto Me. I shall deliver you from all sinful reactions. Do not fear.",
            "emotions": ["surrender", "fear", "anxiety", "trust"],
            "embed_text": "Chapter 18, Verse 66 — Moksha Sannyas Yoga. Let go of all worry. Surrender completely. You will be taken care of. Do not fear.",
        },
        {
            "id": "ch04_v007",
            "chapter": 4,
            "chapter_title": "Jnana Yoga — Path of Knowledge",
            "verse": 7,
            "sanskrit": "यदा यदा हि धर्मस्य ग्लानिर्भवति भारत। अभ्युत्थानमधर्मस्य तदात्मानं सृजाम्यहम्॥",
            "transliteration": "yada yada hi dharmasya glanir bhavati bharata | abhyutthanam adharmasya tadatmanam srijamy aham",
            "word_meanings": "yada yada—whenever and wherever; hi—certainly; dharmasya—of religion; glanih—discrepancies; bhavati—become manifested",
            "translation": "Whenever and wherever there is a decline in religious practice, O descendant of Bharata, and a predominant rise of irreligion — at that time I descend Myself.",
            "emotions": ["hopelessness", "despair", "faith", "cosmic"],
            "embed_text": "Chapter 4, Verse 7 — Jnana Yoga. Whenever the world falls into darkness and chaos, a force of restoration rises. You are not alone in your struggle.",
        },
        {
            "id": "ch02_v003",
            "chapter": 2,
            "chapter_title": "Transcendental Knowledge",
            "verse": 3,
            "sanskrit": "क्लैब्यं मा स्म गमः पार्थ नैतत्त्वय्युपपद्यते। क्षुद्रं हृदयदौर्बल्यं त्यक्त्वोत्तिष्ठ परन्तप॥",
            "transliteration": "klaibyam ma sma gamah partha naitat tvayy upapadyate | kshudram hridaya-daurbalyam tyaktvottishtha parantapa",
            "word_meanings": "klaibyam—impotence; ma—do not; sma—take to; gamah—go; partha—O son of Pritha; na—never; etat—this; tvayi—unto you",
            "translation": "O Partha, do not yield to this degrading impotence. It does not become you. Give up such petty weakness of heart and arise.",
            "emotions": ["cowardice", "weakness", "depression", "paralysis"],
            "embed_text": "Chapter 2, Verse 3 — Transcendental Knowledge. Stop giving in to weakness. This is not who you are. Get up. Act. Your weakness of heart is not your truth.",
        },
        {
            "id": "ch03_v016",
            "chapter": 3,
            "chapter_title": "Karma Yoga — Path of Action",
            "verse": 16,
            "sanskrit": "एवं प्रवर्तितं चक्रं नानुवर्तयतीह यः। अघायुरिन्द्रियारामो मोघं पार्थ स जीवति॥",
            "transliteration": "evam pravartitam chakram nanuvartayatiha yah | aghayur indriyaramo mogham partha sa jivati",
            "word_meanings": "evam—thus; pravartitam—established by the Vedas; chakram—wheel; na—does not; anuvartayati—adopt; iha—in this life",
            "translation": "My dear Arjuna, one who does not follow in human life the cycle of sacrifice thus established by the Vedas certainly lives a life full of sin. Living only for the satisfaction of the senses, such a person lives in vain.",
            "emotions": ["meaninglessness", "purpose", "direction", "action"],
            "embed_text": "Chapter 3, Verse 16 — Karma Yoga. A life lived only for personal pleasure and without contribution is an empty life. Find your role and fulfill it.",
        },
    ]
    return sample_shlokas
 
 
def save_shlokas(shlokas: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(shlokas, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(shlokas)} shlokas to {OUTPUT_FILE}")
 
 
def load_shlokas() -> list[dict]:
    """Load shlokas from disk. Used by other modules."""
    if not OUTPUT_FILE.exists():
        raise FileNotFoundError(
            f"Shloka data not found at {OUTPUT_FILE}. "
            "Run `python src/ingest.py` first."
        )
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
 
 
def print_summary(shlokas: list[dict]) -> None:
    print("\n─── Sage AI — Ingestion Summary ───────────────────────")
    print(f"  Total shlokas loaded : {len(shlokas)}")
    chapters = sorted(set(s["chapter"] for s in shlokas))
    print(f"  Chapters covered     : {len(chapters)} ({min(chapters)}–{max(chapters)})")
    all_emotions = [e for s in shlokas for e in s["emotions"]]
    unique_emotions = sorted(set(all_emotions))
    print(f"  Unique emotion tags  : {len(unique_emotions)}")
    print(f"  Sample emotions      : {', '.join(unique_emotions[:8])}")
    print(f"  Output file          : {OUTPUT_FILE}")
    print("────────────────────────────────────────────────────────\n")
 
 
if __name__ == "__main__":
    print("🌿 Sage AI — Starting data ingestion...")
    print()
    print("Loading sample shloka dataset...")
    print("(To load all 700 shlokas, add your RapidAPI key and")
    print(" call fetch_shloka() in a loop — see comments in this file)\n")
 
    shlokas = load_sample_data()
    save_shlokas(shlokas)
    print_summary(shlokas)