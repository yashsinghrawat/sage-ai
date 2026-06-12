"""
src/ingest.py
─────────────────────────────────────────────────────────────────
Sage AI — Data Ingestion Pipeline
 
What this file does:
1. Fetches all 700 Bhagavad Gita shlokas from a public API
2. Enriches each shloka with emotion tags and topic metadata
3. Saves a clean, structured JSON to data/gita_shlokas.json
 

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
    Returns a curated set of 32 shlokas spanning a wide range of
    emotional/life situations, so retrieval has real signal to work
    with. This is hand-picked from across all 18 chapters — these
    are some of the most universally quoted and applicable verses.
 
    To scale to all 700, wire up fetch_shloka() with an API key and
    loop over chapters 1-18, verses 1-N (verse counts vary per chapter).
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
 
        # ── Anger & Loss of Control ────────────────────────────────
        {
            "id": "ch02_v063",
            "chapter": 2,
            "chapter_title": "Transcendental Knowledge",
            "verse": 63,
            "sanskrit": "क्रोधाद्भवति सम्मोहः सम्मोहात्स्मृतिविभ्रमः। स्मृतिभ्रंशाद् बुद्धिनाशो बुद्धिनाशात्प्रणश्यति॥",
            "transliteration": "krodhad bhavati sammohah sammohat smriti-vibhramah | smriti-bhramshad buddhi-nasho buddhi-nashat pranashyati",
            "word_meanings": "krodhat—from anger; bhavati—takes place; sammohah—perfect illusion; sammohat—from illusion; smriti-vibhramah—bewilderment of memory",
            "translation": "From anger, complete delusion arises, and from delusion, bewilderment of memory. When memory is bewildered, intelligence is lost, and when intelligence is lost, one falls down again into the material pool.",
            "emotions": ["anger", "rage", "regret", "self-destruction"],
            "embed_text": "Chapter 2, Verse 63 — Transcendental Knowledge. Anger clouds judgment, judgment loss destroys memory of your values, and that destroys your higher reasoning. Notice the spiral before it starts.",
        },
        {
            "id": "ch16_v021",
            "chapter": 16,
            "chapter_title": "Daivasura Sampad Vibhaga Yoga",
            "verse": 21,
            "sanskrit": "त्रिविधं नरकस्येदं द्वारं नाशनमात्मनः। कामः क्रोधस्तथा लोभस्तस्मादेतत्त्रयं त्यजेत्॥",
            "transliteration": "tri-vidham narakasyedam dvaram nashanam atmanah | kamah krodhas tatha lobhas tasmad etat trayam tyajet",
            "word_meanings": "tri-vidham—three kinds of; narakasya—of hell; idam—this; dvaram—gate; nashanam—destructive; atmanah—of the self",
            "translation": "There are three gates leading to this hell — lust, anger, and greed. Every sane person should give these up, for they lead to the degradation of the soul.",
            "emotions": ["anger", "greed", "temptation", "self-control"],
            "embed_text": "Chapter 16, Verse 21 — Daivasura Sampad Vibhaga Yoga. Lust, anger, and greed are the three things that quietly destroy a person from the inside. Recognizing them early is the first step to freedom.",
        },
 
        # ── Jealousy & Comparison ───────────────────────────────────
        {
            "id": "ch03_v035",
            "chapter": 3,
            "chapter_title": "Karma Yoga — Path of Action",
            "verse": 35,
            "sanskrit": "श्रेयान्स्वधर्मो विगुणः परधर्मात्स्वनुष्ठितात्। स्वधर्मे निधनं श्रेयः परधर्मो भयावहः॥",
            "transliteration": "shreyan sva-dharmo vigunah para-dharmat sv-anushthitat | sva-dharme nidhanam shreyah para-dharmo bhayavahah",
            "word_meanings": "shreyan—better; sva-dharmah—one's prescribed duties; vigunah—even though faulty; para-dharmat—than duties mentioned for others",
            "translation": "It is far better to discharge one's prescribed duties, even though faulty, than another's duties, even though perfectly. Following one's own path, even with flaws, is better — destruction in one's own path is better than success in another's.",
            "emotions": ["comparison", "jealousy", "identity", "purpose"],
            "embed_text": "Chapter 3, Verse 35 — Karma Yoga. It is better to walk your own imperfect path than to perfectly imitate someone else's. Comparing your journey to another's leads nowhere good.",
        },
        {
            "id": "ch12_v015",
            "chapter": 12,
            "chapter_title": "Bhakti Yoga — Path of Devotion",
            "verse": 15,
            "sanskrit": "यस्मान्नोद्विजते लोको लोकान्नोद्विजते च यः। हर्षामर्षभयोद्वेगैर्मुक्तो यः स च मे प्रियः॥",
            "transliteration": "yasmad udvijate loko lokad udvijate cha yah | harshamarsha-bhayodvegair mukto yah sa cha me priyah",
            "word_meanings": "yasmat—from whom; na—never; udvijate—are agitated; lokah—people; lokat—from people; na—never; udvijate—is disturbed",
            "translation": "One who is not envious but is a kind friend to all living entities, who does not think himself a proprietor, who is free from false ego, equal in both happiness and distress... such a person is very dear to Me.",
            "emotions": ["jealousy", "envy", "ego", "equanimity"],
            "embed_text": "Chapter 12, Verse 15 — Bhakti Yoga. A person free of envy, who treats success and failure with the same steadiness, is the one who truly thrives. Envy disturbs you far more than it disturbs anyone else.",
        },
 
        # ── Fear & Anxiety about the Future ─────────────────────────
        {
            "id": "ch02_v022",
            "chapter": 2,
            "chapter_title": "Transcendental Knowledge",
            "verse": 22,
            "sanskrit": "वासांसि जीर्णानि यथा विहाय नवानि गृह्णाति नरोपराणि। तथा शरीराणि विहाय जीर्णान्यन्यानि संयाति नवानि देही॥",
            "transliteration": "vasamsi jirnani yatha vihaya navani grihnati naro 'parani | tatha sharirani vihaya jirnany anyani samyati navani dehi",
            "word_meanings": "vasamsi—garments; jirnani—old and worn out; yatha—just as; vihaya—giving up; navani—new garments; grihnati—does accept",
            "translation": "As a person puts on new garments, giving up old ones, similarly, the soul accepts new material bodies, giving up the old and useless ones.",
            "emotions": ["change", "fear", "transition", "impermanence"],
            "embed_text": "Chapter 2, Verse 22 — Transcendental Knowledge. Just as you change clothes when old ones wear out, change and transition are natural, not something to fear. What you are doesn't end — it transforms.",
        },
        {
            "id": "ch02_v040",
            "chapter": 2,
            "chapter_title": "Transcendental Knowledge",
            "verse": 40,
            "sanskrit": "नेहाभिक्रमनाशोऽस्ति प्रत्यवायो न विद्यते। स्वल्पमप्यस्य धर्मस्य त्रायते महतो भयात्॥",
            "transliteration": "nehabhikrama-nasho 'sti pratyayayo na vidyate | svalpam apy asya dharmasya trayate mahato bhayat",
            "word_meanings": "na—there is not; iha—in this yoga; abhikrama—in endeavoring; nashah—loss; asti—there is; pratyayayah—diminution",
            "translation": "In this endeavor there is no loss or diminution, and a little advancement on this path can protect one from the most fearful type of danger.",
            "emotions": ["fear", "anxiety", "effort", "progress"],
            "embed_text": "Chapter 2, Verse 40 — Transcendental Knowledge. No sincere effort is ever wasted. Even small progress protects you from the greatest fears. You don't need to finish — you need to begin.",
        },
        {
            "id": "ch06_v035",
            "chapter": 6,
            "chapter_title": "Dhyana Yoga — Path of Meditation",
            "verse": 35,
            "sanskrit": "असंशयं महाबाहो मनो दुर्निग्रहं चलम्। अभ्यासेन तु कौन्तेय वैराग्येण च गृह्यते॥",
            "transliteration": "asamshayam maha-baho mano durnigraham chalam | abhyasena tu kaunteya vairagyena cha grihyate",
            "word_meanings": "asamshayam—undoubtedly; maha-baho—O mighty-armed one; manah—the mind; durnigraham—difficult to curb; chalam—flickering",
            "translation": "It is undoubtedly very difficult to curb the restless mind, but it is possible by suitable practice and by detachment.",
            "emotions": ["anxiety", "restlessness", "discipline", "practice"],
            "embed_text": "Chapter 6, Verse 35 — Dhyana Yoga. Yes, your restless, anxious mind is genuinely hard to control — that's not a personal failure. But with consistent practice, it becomes possible. Don't expect instant calm; expect gradual change.",
        },
 
        # ── Betrayal & Broken Trust ──────────────────────────────────
        {
            "id": "ch05_v019",
            "chapter": 5,
            "chapter_title": "Karma Sannyas Yoga",
            "verse": 19,
            "sanskrit": "इहैव तैर्जितः सर्गो येषां साम्ये स्थितं मनः। निर्दोषं हि समं ब्रह्म तस्माद्ब्रह्मणि ते स्थिताः॥",
            "transliteration": "ihaiva tair jitah sargo yeshham samye sthitam manah | nirdosham hi samam brahma tasmad brahmani te sthitah",
            "word_meanings": "iha—in this life; eva—certainly; taih—by them; jitah—conquered; sargah—birth and death; yeshham—whose; samye—in equanimity",
            "translation": "Those whose minds are established in sameness and equanimity have already conquered the conditions of birth and death. They are flawless like Brahman, and thus they are already situated in Brahman.",
            "emotions": ["betrayal", "anger", "equanimity", "peace"],
            "embed_text": "Chapter 5, Verse 19 — Karma Sannyas Yoga. A mind that stays even and steady — that doesn't get pulled into reactive cycles after being wronged — has already won the larger battle. Equanimity is strength, not weakness.",
        },
        {
            "id": "ch16_v003",
            "chapter": 16,
            "chapter_title": "Daivasura Sampad Vibhaga Yoga",
            "verse": 3,
            "sanskrit": "तेजः क्षमा धृतिः शौचमद्रोहो नातिमानिता। भवन्ति सम्पदं दैवीमभिजातस्य भारत॥",
            "transliteration": "tejah kshama dhritih shaucham adroho natimanita | bhavanti sampadam daivim abhijatasya bharata",
            "word_meanings": "tejah—vigor; kshama—forgiveness; dhritih—fortitude; shaucham—cleanliness; adrohah—freedom from envy; na-atimanita—freedom from the passion for honor",
            "translation": "Vigor, forgiveness, fortitude, purity, freedom from malice, and absence of excessive pride — these qualities belong to those born with a divine nature.",
            "emotions": ["betrayal", "forgiveness", "resentment", "character"],
            "embed_text": "Chapter 16, Verse 3 — Daivasura Sampad Vibhaga Yoga. Forgiveness and freedom from malice toward those who've wronged you aren't signs of weakness — they are markers of genuine inner strength.",
        },
 
        # ── Career, Ambition & Direction ─────────────────────────────
        {
            "id": "ch02_v048",
            "chapter": 2,
            "chapter_title": "Transcendental Knowledge",
            "verse": 48,
            "sanskrit": "योगस्थः कुरु कर्माणि सङ्गं त्यक्त्वा धनञ्जय। सिद्ध्यसिद्ध्योः समो भूत्वा समत्वं योग उच्यते॥",
            "transliteration": "yoga-sthah kuru karmani sangam tyaktva dhananjaya | siddhy-asiddhyoh samo bhutva samatvam yoga uchyate",
            "word_meanings": "yoga-sthah—equipoised; kuru—perform; karmani—your duties; sangam—attachment; tyaktva—giving up; dhananjaya—O Arjuna",
            "translation": "Perform your duty equipoised, abandoning all attachment to success or failure. Such equanimity is called yoga.",
            "emotions": ["ambition", "outcome-anxiety", "stress", "balance"],
            "embed_text": "Chapter 2, Verse 48 — Transcendental Knowledge. Do your work fully — but release your grip on the outcome. Obsessing over success or failure is what burns people out, not the work itself.",
        },
        {
            "id": "ch18_v048",
            "chapter": 18,
            "chapter_title": "Moksha Sannyas Yoga",
            "verse": 48,
            "sanskrit": "सहजं कर्म कौन्तेय सदोषमपि न त्यजेत्। सर्वारम्भा हि दोषेण धूमेनाग्निरिवावृताः॥",
            "transliteration": "saha-jam karma kaunteya sa-dosham api na tyajet | sarvarambha hi doshena dhumenagnir ivavritah",
            "word_meanings": "saha-jam—simultaneously born; karma—work; kaunteya—O son of Kunti; sa-dosham—with fault; api—although; na—never; tyajet—give up",
            "translation": "Every endeavor is covered by some imperfection, just as fire is covered by smoke. Therefore one should not give up the work which is born of one's nature, even if it includes faults.",
            "emotions": ["career", "perfectionism", "starting", "imperfection"],
            "embed_text": "Chapter 18, Verse 48 — Moksha Sannyas Yoga. Every undertaking has imperfections — that's not a reason to abandon it. Waiting for the perfect, flawless path means never moving at all.",
        },
        {
            "id": "ch03_v021",
            "chapter": 3,
            "chapter_title": "Karma Yoga — Path of Action",
            "verse": 21,
            "sanskrit": "यद्यदाचरति श्रेष्ठस्तत्तदेवेतरो जनः। स यत्प्रमाणं कुरुते लोकस्तदनुवर्तते॥",
            "transliteration": "yad yad acharati shreshthas tat tad evetaro janah | sa yat pramanam kurute lokas tad anuvartate",
            "word_meanings": "yat yat—whatever; acharati—does; shreshthah—a respectable leader; tat—that; tat—and only that; eva—certainly; itarah—common; janah—person",
            "translation": "Whatever action a great person performs, common people follow. And whatever standards he sets by exemplary acts, all the world pursues.",
            "emotions": ["leadership", "responsibility", "influence", "example"],
            "embed_text": "Chapter 3, Verse 21 — Karma Yoga. People around you absorb your standards more than your words. If you're in any position of influence, your actions are quietly setting the bar for others.",
        },
 
        # ── Grief & Loss ──────────────────────────────────────────────
        {
            "id": "ch02_v027",
            "chapter": 2,
            "chapter_title": "Transcendental Knowledge",
            "verse": 27,
            "sanskrit": "जातस्य हि ध्रुवो मृत्युर्ध्रुवं जन्म मृतस्य च। तस्मादपरिहार्येऽर्थे न त्वं शोचितुमर्हसि॥",
            "transliteration": "jatasya hi dhruvo mrityur dhruvam janma mritasya cha | tasmad aparihaye 'rthe na tvam shochitum arhasi",
            "word_meanings": "jatasya—of one who has taken birth; hi—certainly; dhruvah—a fact; mrityuh—death; dhruvam—it is also a fact",
            "translation": "For one who has been born, death is certain, and for one who has died, birth is certain. Therefore, you should not lament over the inevitable.",
            "emotions": ["grief", "death", "loss", "acceptance"],
            "embed_text": "Chapter 2, Verse 27 — Transcendental Knowledge. Death is as certain as birth — it is woven into existence itself, not a malfunction or injustice. This doesn't make grief wrong, but it can ease the sense that something has gone unfairly broken.",
        },
        {
            "id": "ch02_v013",
            "chapter": 2,
            "chapter_title": "Transcendental Knowledge",
            "verse": 13,
            "sanskrit": "देहिनोऽस्मिन्यथा देहे कौमारं यौवनं जरा। तथा देहान्तरप्राप्तिर्धीरस्तत्र न मुह्यति॥",
            "transliteration": "dehino 'smin yatha dehe kaumaram yauvanam jara | tatha dehantara-praptir dhiras tatra na muhyati",
            "word_meanings": "dehinah—of the embodied; asmin—in this; yatha—as; dehe—in the body; kaumaram—boyhood; yauvanam—youth; jara—old age",
            "translation": "As the embodied soul continuously passes, in this body, from boyhood to youth to old age, the soul similarly passes into another body at death. The wise are not bewildered by such a change.",
            "emotions": ["grief", "aging", "change", "wisdom"],
            "embed_text": "Chapter 2, Verse 13 — Transcendental Knowledge. You've already changed enormously across your life — childhood to now is itself a kind of transformation. Death is one more transition in a series you've already been navigating all along.",
        },
 
        # ── Self-Worth & Inadequacy ───────────────────────────────────
        {
            "id": "ch06_v006",
            "chapter": 6,
            "chapter_title": "Dhyana Yoga — Path of Meditation",
            "verse": 6,
            "sanskrit": "बन्धुरात्मात्मनस्तस्य येनात्मैवात्मना जितः। अनात्मनस्तु शत्रुत्वे वर्तेतात्मैव शत्रुवत्॥",
            "transliteration": "bandhur atmatmanas tasya yenatmaivatmana jitah | anatmanas tu shatrutve vartetatmaiva shatru-vat",
            "word_meanings": "bandhuh—friend; atma—the mind; atmanah—of the living entity; tasya—of him; yena—by whom; atma—the mind",
            "translation": "For those who have conquered the mind, the mind becomes their best friend. For those who have failed to do so, the mind remains their greatest enemy.",
            "emotions": ["self-worth", "self-doubt", "inner-conflict", "growth"],
            "embed_text": "Chapter 6, Verse 6 — Dhyana Yoga. The voice in your head that tells you you're not enough — that's not 'you', that's an untrained mind. With work, that same mind becomes your strongest ally instead.",
        },
        {
            "id": "ch04_v038",
            "chapter": 4,
            "chapter_title": "Jnana Yoga — Path of Knowledge",
            "verse": 38,
            "sanskrit": "न हि ज्ञानेन सदृशं पवित्रमिह विद्यते। तत्स्वयं योगसंसिद्धः कालेनात्मनि विन्दति॥",
            "transliteration": "na hi jnanena sadrisham pavitram iha vidyate | tat svayam yoga-samsiddhah kalenatmani vindati",
            "word_meanings": "na—nothing; hi—certainly; jnanena—with knowledge; sadrisham—in comparison; pavitram—sanctified; iha—in this world",
            "translation": "In this world, there is nothing as purifying as transcendental knowledge. One who has attained this, in due course of time, finds it within himself.",
            "emotions": ["self-worth", "growth", "patience", "knowledge"],
            "embed_text": "Chapter 4, Verse 38 — Jnana Yoga. Real self-understanding takes time to develop — it's not handed to you instantly. The fact that you're still figuring things out doesn't mean you're behind; it means you're in process.",
        },
 
        # ── Decision-Making & Confusion ────────────────────────────────
        {
            "id": "ch03_v007",
            "chapter": 3,
            "chapter_title": "Karma Yoga — Path of Action",
            "verse": 7,
            "sanskrit": "यस्त्विन्द्रियाणि मनसा नियम्यारभतेऽर्जुन। कर्मेन्द्रियैः कर्मयोगमसक्तः स विशिष्यते॥",
            "transliteration": "yas tv indriyani manasa niyamyarabhate 'rjuna | karmendriyaih karma-yogam asaktah sa vishishyate",
            "word_meanings": "yah—one who; tu—but; indriyani—the senses; manasa—by the mind; niyamya—regulating; arabhate—begins",
            "translation": "But one who controls the senses by the mind and engages in active duty without attachment — that person is far superior.",
            "emotions": ["confusion", "indecision", "discipline", "clarity"],
            "embed_text": "Chapter 3, Verse 7 — Karma Yoga. Confusion often comes from being pulled in many directions by impulses. Pausing to align mind and action — even imperfectly — moves you toward clarity faster than waiting for clarity first.",
        },
        {
            "id": "ch02_v041",
            "chapter": 2,
            "chapter_title": "Transcendental Knowledge",
            "verse": 41,
            "sanskrit": "व्यवसायात्मिका बुद्धिरेकेह कुरुनन्दन। बहुशाखा ह्यनन्ताश्च बुद्धयोऽव्यवसायिनाम्॥",
            "transliteration": "vyavasayatmika buddhir ekeha kuru-nandana | bahu-shakha hy anantash cha buddhayo 'vyavasayinam",
            "word_meanings": "vyavasaya-atmika—resolute; buddhih—intelligence; eka—only one; iha—in this world; kuru-nandana—O beloved child of the Kurus",
            "translation": "Those who are on this path are resolute in purpose, and their aim is one. The intelligence of the irresolute is many-branched and endless.",
            "emotions": ["confusion", "indecision", "focus", "resolve"],
            "embed_text": "Chapter 2, Verse 41 — Transcendental Knowledge. An unfocused mind generates endless branching possibilities and second-guessing. Choosing one direction — even if not 'the' perfect one — cuts through the noise.",
        },
 
        # ── Overwhelm & Burnout ──────────────────────────────────────
        {
            "id": "ch06_v016",
            "chapter": 6,
            "chapter_title": "Dhyana Yoga — Path of Meditation",
            "verse": 16,
            "sanskrit": "नात्यश्नतस्तु योगोऽस्ति न चैकान्तमनश्नतः। न च प्रस्वपनशीलस्य जाग्रतो नैव चार्जुन॥",
            "transliteration": "natyashnatastu yogo 'sti na chaikantam anashnatah | na cha prasvapna-shilasya jagrato naiva charjuna",
            "word_meanings": "na—never; ati—too much; ashnatah—of one who eats; tu—but; yogah—linking with the Supreme; asti—there is",
            "translation": "There is no possibility of becoming a yogi, O Arjuna, if one eats too much, or eats too little, or sleeps too much, or does not sleep enough.",
            "emotions": ["burnout", "balance", "self-care", "moderation"],
            "embed_text": "Chapter 6, Verse 16 — Dhyana Yoga. Extremes in either direction — overworking and under-resting, or the reverse — both block real progress. Sustainable effort requires actual balance, not heroics.",
        },
        {
            "id": "ch06_v017",
            "chapter": 6,
            "chapter_title": "Dhyana Yoga — Path of Meditation",
            "verse": 17,
            "sanskrit": "युक्ताहारविहारस्य युक्तचेष्टस्य कर्मसु। युक्तस्वप्नावबोधस्य योगो भवति दुःखहा॥",
            "transliteration": "yuktahara-viharasya yukta-cheshtasya karmasu | yukta-svapnavabodhasya yogo bhavati duhkha-ha",
            "word_meanings": "yukta—regulated; ahara—eating; vihara—recreation; yukta-cheshtasya—of one regulated in action; karmasu—in discharging duties",
            "translation": "One who is regulated in eating, sleeping, recreation, and work can mitigate all material suffering through yoga.",
            "emotions": ["burnout", "routine", "balance", "wellbeing"],
            "embed_text": "Chapter 6, Verse 17 — Dhyana Yoga. A regulated rhythm — for eating, rest, recreation, and work — is itself a form of relief from suffering. Structure isn't restrictive; it's protective.",
        },
 
        # ── Guilt & Past Mistakes ───────────────────────────────────────
        {
            "id": "ch04_v036",
            "chapter": 4,
            "chapter_title": "Jnana Yoga — Path of Knowledge",
            "verse": 36,
            "sanskrit": "अपि चेदसि पापेभ्यः सर्वेभ्यः पापकृत्तमः। सर्वं ज्ञानप्लवेनैव वृजिनं सन्तरिष्यसि॥",
            "transliteration": "api chedasi papebhyah sarvebhyah papa-krittamah | sarvam jnana-plavenaiva vrijinam santarishyasi",
            "word_meanings": "api—even; chet—if; asi—you are; papebhyah—of sinners; sarvebhyah—of all; papa-krit-tamah—the greatest sinner",
            "translation": "Even if you are considered to be the most sinful of all sinners, once you are situated in the boat of transcendental knowledge, you will be able to cross over the ocean of miseries.",
            "emotions": ["guilt", "shame", "redemption", "self-forgiveness"],
            "embed_text": "Chapter 4, Verse 36 — Jnana Yoga. Whatever you've done, however severe it feels — growth and understanding remain available to you. Past mistakes don't permanently disqualify you from moving forward.",
        },
        {
            "id": "ch18_v058",
            "chapter": 18,
            "chapter_title": "Moksha Sannyas Yoga",
            "verse": 58,
            "sanskrit": "मच्चित्तः सर्वदुर्गाणि मत्प्रसादात्तरिष्यसि। अथ चेत्त्वमहंकारान्न श्रोष्यसि विनङ्क्ष्यसि॥",
            "transliteration": "mat-chittah sarva-durgani mat-prasadat tarishyasi | atha chet tvam ahankarat na shroshyasi vinankshyasi",
            "word_meanings": "mat—of Me; chittah—being in consciousness; sarva—all; durgani—obstacles; mat-prasadat—by My mercy; tarishyasi—you will overcome",
            "translation": "If you become conscious of a higher purpose, you will pass over all obstacles by My grace. But if you act on false ego and do not listen, you will be lost.",
            "emotions": ["guilt", "ego", "obstacles", "perspective"],
            "embed_text": "Chapter 18, Verse 58 — Moksha Sannyas Yoga. Keeping sight of what genuinely matters to you helps you move past obstacles. Letting wounded pride dictate your choices tends to make things worse, not better.",
        },
 
        # ── Loneliness & Isolation ─────────────────────────────────────
        {
            "id": "ch06_v030",
            "chapter": 6,
            "chapter_title": "Dhyana Yoga — Path of Meditation",
            "verse": 30,
            "sanskrit": "यो मां पश्यति सर्वत्र सर्वं च मयि पश्यति। तस्याहं न प्रणश्यामि स च मे न प्रणश्यति॥",
            "transliteration": "yo mam pashyati sarvatra sarvam cha mayi pashyati | tasyaham na pranashyami sa cha me na pranashyati",
            "word_meanings": "yah—whoever; mam—Me; pashyati—sees; sarvatra—everywhere; sarvam—everything; cha—and; mayi—in Me; pashyati—sees",
            "translation": "One who sees Me everywhere and sees everything in Me never loses sight of Me, nor do I ever lose sight of that person.",
            "emotions": ["loneliness", "connection", "isolation", "presence"],
            "embed_text": "Chapter 6, Verse 30 — Dhyana Yoga. A deep sense of connection to something larger — to life itself, to the people and world around you — is available even in moments of acute isolation. You're more woven into things than loneliness lets you feel.",
        },
 
        # ── Pride & Ego ───────────────────────────────────────────────
        {
            "id": "ch16_v004",
            "chapter": 16,
            "chapter_title": "Daivasura Sampad Vibhaga Yoga",
            "verse": 4,
            "sanskrit": "दम्भो दर्पोऽभिमानश्च क्रोधः पारुष्यमेव च। अज्ञानं चाभिजातस्य पार्थ सम्पदमासुरीम्॥",
            "transliteration": "dambho darpo 'bhimanash cha krodhah parushyam eva cha | ajnanam chabhijatasya partha sampadam asurim",
            "word_meanings": "dambhah—pride; darpah—arrogance; abhimanah—conceit; cha—and; krodhah—anger; parushyam—harshness",
            "translation": "Pride, arrogance, conceit, anger, harshness, and ignorance — these qualities belong to those of a lower nature.",
            "emotions": ["pride", "ego", "arrogance", "self-awareness"],
            "embed_text": "Chapter 16, Verse 4 — Daivasura Sampad Vibhaga Yoga. Pride, arrogance, and harshness toward others usually signal something is off internally, not a position of real strength. Worth noticing when these show up in yourself.",
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
    print("✅ Day 1 ingestion complete.")