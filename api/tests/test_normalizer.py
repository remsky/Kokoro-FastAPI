"""Tests for text normalization service"""

import pytest

from api.src.services.text_processing.normalization import normalize_text
from api.src.structures.schemas import NormalizationOptions

CASES = [
    # url protocols (pr #12)
    ("Check out https://example.com", "Check out https example dot com"),
    ("Visit http://site.com", "Visit http site dot com"),
    ("Go to https://test.org/path", "Go to https test dot org slash path"),
    # url www (pr #10, #12)
    ("Go to www.example.com", "Go to www example dot com"),
    ("Visit www.test.org/docs", "Visit www test dot org slash docs"),
    ("Check www.site.com?q=test", "Check www site dot com question-mark q equals test"),
    # url localhost (pr #303)
    ("Running on localhost:7860", "Running on localhost colon seventy-eight sixty"),
    (
        "Server at localhost:8080/api",
        "Server at localhost colon eighty eighty slash api",
    ),
    (
        "Test localhost:3000/test?v=1",
        "Test localhost colon three thousand slash test question-mark v equals one",
    ),
    # url ip addresses (pr #303)
    (
        "Access 0.0.0.0:9090/test",
        "Access zero dot zero dot zero dot zero colon ninety ninety slash test",
    ),
    (
        "API at 192.168.1.1:8000",
        "API at one hundred and ninety-two dot one hundred and sixty-eight dot one dot one colon eight thousand",
    ),
    (
        "Server 127.0.0.1",
        "Server one hundred and twenty-seven dot zero dot zero dot one",
    ),
    # url raw domains (pr #12)
    ("Visit google.com/search", "Visit google dot com slash search"),
    (
        "Go to example.com/path?q=test",
        "Go to example dot com slash path question-mark q equals test",
    ),
    ("Check docs.test.com", "Check docs dot test dot com"),
    # email (pr #10, #12)
    ("Email me at user@example.com", "Email me at user at example dot com"),
    ("Contact admin@test.org", "Contact admin at test dot org"),
    ("Send to test.user@site.com", "Send to test dot user at site dot com"),
    # money (pr #256, #303)
    ("He lost $5.3 thousand.", "He lost five point three thousand dollars."),
    (
        "He went gambling and lost about $25.05k.",
        "He went gambling and lost about twenty-five point zero five thousand dollars.",
    ),
    (
        "To put it weirdly -$6.9 million",
        "To put it weirdly minus six point nine million dollars",
    ),
    ("It costs $50.3.", "It costs fifty dollars and thirty cents."),
    (
        "The plant cost $200,000.8.",
        "The plant cost two hundred thousand dollars and eighty cents.",
    ),
    (
        "Your shopping spree cost $674.03!",
        "Your shopping spree cost six hundred and seventy-four dollars and three cents!",
    ),
    ("€30.2 is in euros", "thirty euros and twenty cents is in euros"),
    ("It costs $50.", "It costs fifty dollars."),
    ("£5.50 please", "five pounds and fifty pence please"),
    ("$1 trillion", "one trillion dollars"),
    ("Costs $1,000,000", "Costs one million dollars"),
    ("-$5", "minus five dollars"),
    # issue #224
    (
        "still slated to lose $400 million.",
        "still slated to lose four hundred million dollars.",
    ),
    (
        "an endowment of $14.8 billion;",
        "an endowment of fourteen point eight billion dollars;",
    ),
    # issue #302
    ("I spent $200,000 on stuff", "I spent two hundred thousand dollars on stuff"),
    # time (pr #303)
    ("Your flight leaves at 10:35 pm", "Your flight leaves at ten thirty-five pm"),
    (
        "He departed for london around 5:03 am.",
        "He departed for london around five oh three am.",
    ),
    (
        "Only the 13:42 and 15:12 slots are available.",
        "Only the thirteen forty-two and fifteen twelve slots are available.",
    ),
    ("It is currently 1:00 pm", "It is currently one pm"),
    ("It is currently 3:00", "It is currently three o'clock"),
    ("12:00 am is midnight", "twelve am is midnight"),
    ("Lap time 1:02:03", "Lap time one oh two and three seconds"),
    ("Finished in 0:05:01", "Finished in zero oh five and one second"),
    ("Logged at 12:30:15 pm", "Logged at twelve thirty and fifteen seconds pm"),
    # numbers (pr #303)
    (
        "I bought 1035 cans of soda",
        "I bought one thousand and thirty-five cans of soda",
    ),
    (
        "The bus has a maximum capacity of 62 people",
        "The bus has a maximum capacity of sixty-two people",
    ),
    (
        "There are 1300 products left in stock",
        "There are one thousand, three hundred products left in stock",
    ),
    (
        "The population is 7,890,000 people.",
        "The population is seven million, eight hundred and ninety thousand people.",
    ),
    (
        "He looked around but only found 1.6k of the 10k bricks",
        "He looked around but only found one point six thousand of the ten thousand bricks",
    ),
    ("The book has 342 pages.", "The book has three hundred and forty-two pages."),
    ("He made -50 sales today.", "He made minus fifty sales today."),
    (
        "56.789 to the power of 1.35 million",
        "fifty-six point seven eight nine to the power of one point three five million",
    ),
    ("Year 1999 and 2024", "Year nineteen ninety-nine and twenty twenty-four"),
    ("in 3497", "in thirty-four ninety-seven"),
    ("Room 1005", "Room one thousand and five"),
    # thousands separators rule out the year reading (#259)
    (
        "3,497 head of cattle",
        "three thousand, four hundred and ninety-seven head of cattle",
    ),
    (
        "$3,497.50",
        "three thousand, four hundred and ninety-seven dollars and fifty cents",
    ),
    ("$99,070.01", "ninety-nine thousand and seventy dollars and one cent"),
    ("the 3,497th item", "the 3,497th item"),
    ("The year 2,024", "The year two thousand and twenty-four"),
    ("Worth $1.5million now", "Worth dollar 1.5million now"),
    ("12,34", "one thousand, two hundred and thirty-four"),
    ("9" * 400, "9" * 400),
    ("9" * 36, "9" * 36),
    ("9" * 40, "9" * 40),
    ("$" + "9" * 36, " dollar " + "9" * 36),
    ("1.1." + "9" * 5000, "1.1." + "9" * 5000),
    ("The 1980S were loud", "The nineteen eighty S were loud"),
    ("It is 0.5 done", "It is zero point five done"),
    # ranges and dotted versions (pr #492)
    ("Pages 10-20", "Pages ten to twenty"),
    ("Version 2.0.1 released", "Version two point zero point one released"),
    ("Python 3.10.12", "Python three point ten point twelve"),
    ("100% sure", "one hundred percent sure"),
    ("Only .5 left", "Only zero point five left"),
    ("Saved as file_1", "Saved as file one"),
    # digits glued to letters are left for the phonemizer
    ("MP3 player", "MP3 player"),
    ("B2B sales", "B2B sales"),
    ("COVID19 era", "COVID19 era"),
    ("COVID-19 era", "COVID-nineteen era"),
    ("1.5x faster", "1.5x faster"),
    ("Try v1.0 now", "Try v1.0 now"),
    ("The 1st and 2nd", "The 1st and 2nd"),
    # units are off by default
    ("It is 10 km away", "It is ten km away"),
    # optional pluralization
    ("Bring your friend(s)", "Bring your friends"),
    # titles and abbreviations
    (
        "Dr. Smith met Mr. Jones and Mrs. Lee and Ms. Kim.",
        "Doctor Smith met Mister Jones and Mrs Lee and Miss Kim.",
    ),
    ("Apples, pears, etc. are fruit", "Apples, pears, etc are fruit"),
    ("The U.S.A. beats all", "The U-S-A- beats all"),
    ("N.A.S.A. launched", "N-A-S-A- launched"),
    ("Yeah, sure. Yea!", "Ye'a, sure. Ye'a!"),
    # "No." stays a word (issue #260)
    (
        "Blinked? No. Smiled ever so slightly? Yes.",
        "Blinked? No. Smiled ever so slightly? Yes.",
    ),
    # phone numbers (pr #179)
    ("Call 555-123-4567", "Call five five five, one two three, four five six seven"),
    ("Call (555) 123-4567", "Call five five five, one two three, four five six seven"),
    (
        "Call +1 555 123 4567",
        "Call one, five five five, one two three, four five six seven",
    ),
    (
        "Dial 555.123.4567 now",
        "Dial five five five, one two three, four five six seven now",
    ),
    # plural and possessive acronyms (pr #493)
    ("the DVD's case", "the DVD's case"),
    ("two CDs and DVDs", "two CDs and DVDs"),
    # punctuation and whitespace
    ("你好，世界", "你好, 世界"),
    ("你好。世界！好吗？再见", "你好. 世界! 好吗? 再见"),
    ("一、二；三：四–五", "一, 二; 三: 四- 五"),
    ("He said “hi” and ‘bye’", "He said \"hi\" and 'bye'"),
    ("Tab\there and   spaces", "Tab here and spaces"),
    ("Line one\n\nLine two", "Line one Line two"),
    # issue #249
    ("What key--the key", "What key — the key"),
    ("The Marbles. --- In", "The Marbles. — In"),
    # em dash passes through (issue #224)
    (
        "a small group)\u2014and at the same time",
        "a small group)\u2014and at the same time",
    ),
    # non-url text
    ("This is not.a.url text", "This is not-a-url text"),
    ("Hello, how are you today?", "Hello, how are you today?"),
    # remaining symbols (pr #322)
    (
        "I love buying products @ good store here & @ other store",
        "I love buying products at good store here and at other store",
    ),
    # 're contractions: wh-words expand so espeak does not voice a spurious "-ray" ending (pr #488)
    (
        "Hello there, how're you doing this fine day?",
        "Hello there, how are you doing this fine day?",
    ),
    (
        "What're these and where're they going?",
        "What are these and where are they going?",
    ),
    (
        "You're sure we're not late and they're here?",
        "You're sure we're not late and they're here?",
    ),
    # apostrophe-less variants use a different word list: "were" and "whore" are real words (pr #493)
    ("Howre you doing and whatre these?", "How are you doing and what are these?"),
    (
        "Youre early and theyre already here.",
        "You are early and they are already here.",
    ),
    ("They were early, and there were more.", "They were early, and there were more."),
    (
        "Therefore the genre was hardcore before more.",
        "Therefore the genre was hardcore before more.",
    ),
    # all-caps names and headers
    ("ARNE SAKNUSSEMM", "arne saknussemm"),
    ("CHAPTER IV. ARNE SAKNUSSEMM", "chapter IV. arne saknussemm"),
    ("LIDENBROCK.", "lidenbrock."),
    ("A TALE OF TWO CITIES", "A tale OF TWO cities"),
    ("the FBI and CIA agreed", "the FBI and CIA agreed"),
    ("US GDP grew", "US GDP grew"),
    ("an HDMI cable", "an HDMI cable"),
    ("NASA JPL", "nasa JPL"),
]

UNITS = NormalizationOptions(unit_normalization=True)

OPTION_CASES = [
    ("It is 10 km away", "It is ten kilometers away", UNITS),
    ("It weighs 1 kg", "It weighs one kilogram", UNITS),
    ("Speed 100 mph", "Speed one hundred miles per hour", UNITS),
    ("The file is 5 MB", "The file is five megabytes", UNITS),
    ("Download 5MB", "Download five megabytes", UNITS),
    ("Set to 10 Mb", "Set to ten megabits", UNITS),
    ("Runs at 3.5 GHz", "Runs at three point five gigahertz", UNITS),
    ("Tuned to 1 Hz", "Tuned to one hertz", UNITS),
    ("Wait 5 min", "Wait five minutes", UNITS),
    ("Wait 1 min", "Wait one minute", UNITS),
    # pr #526
    ("It is 72 F outside", "It is seventy-two degrees fahrenheit outside", UNITS),
    ("It is 72 °F outside", "It is seventy-two degrees fahrenheit outside", UNITS),
    (
        "Visit www.example.com",
        "Visit www-example-com",
        NormalizationOptions(url_normalization=False),
    ),
    (
        "Email user@example.com",
        "Email user at example dot com",
        NormalizationOptions(email_normalization=False),
    ),
    (
        "Call 555-123-4567",
        "Call five hundred and fifty-five to one hundred and twenty-three to forty-five sixty-seven",
        NormalizationOptions(phone_normalization=False),
    ),
    (
        "Bring your friend(s)",
        "Bring your friend(s)",
        NormalizationOptions(optional_pluralization_normalization=False),
    ),
    ("a & b @ c", "a & b @ c", NormalizationOptions(replace_remaining_symbols=False)),
    (
        "ARNE SAKNUSSEMM",
        "ARNE SAKNUSSEMM",
        NormalizationOptions(caps_normalization=False),
    ),
]


@pytest.mark.parametrize("text, expected", CASES, ids=[text for text, _ in CASES])
def test_normalize(text, expected):
    assert normalize_text(text, NormalizationOptions()) == expected


@pytest.mark.parametrize(
    "text, expected, options", OPTION_CASES, ids=[text for text, _, _ in OPTION_CASES]
)
def test_normalize_with_options(text, expected, options):
    assert normalize_text(text, options) == expected
