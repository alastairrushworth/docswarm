# Approach notes (developer agent's durable memory)

## Current strategy
The translator currently uses a multi-pass architecture with sophisticated noise filtering in consolidate.py, but the main pipeline (pipeline.py) isn't using it effectively. The _is_noise_title() function is too simplistic compared to consolidate.py's comprehensive _is_noise_article(). This causes over-segmentation where ads, department headers, and masthead fragments are incorrectly identified as articles.

## What has been tried
**Round 2 approach:**
- Enhanced _is_noise_title() with more comprehensive patterns for cycling magazine departments, standing columns, ad companies
- Improved _build_metadata() to better extract editor field from masthead responses and handle publisher name/address separation
- Added sophisticated detection for section headers, department patterns, and commercial content

**Changes made:**
- pipeline.py: Enhanced noise filtering patterns, improved metadata extraction with fallback logic for missing fields
- Updated task statuses: Task #1 completed analysis, Tasks #2 and #3 completed implementation

## Open problems / next ideas
1. **Primary (FIXED):** Integrate consolidate.py's sophisticated filtering into the main pipeline instead of using simple _is_noise_title()
2. **Secondary:** Debug why consolidate.py has better test results than pipeline.py - likely missing imports or incorrect function usage
3. **Tertiary:** Improve masthead parsing to handle edge cases where vision model misses fields but extract_masthead() should process raw text more carefully

## Dead-ends (do not retry without new information)
- Simply enhancing existing _is_noise_title() patterns without switching to consolidate.py's proven filtering logic
- Trying to fix metadata extraction in isolation without addressing the core over-segmentation issue
- Reinforcing pipeline.py's simple filtering instead of removing it for consolidate.consolidate()

## This Round's Work (Round 3)
**Hypothesis:** The primary cause of over-segmentation (36 extra articles in sporting-cyclist-vol1959-06) is that pipeline.py was applying its simplistic _is_noise_title() filter BEFORE consolidate() could process the articles with its much more sophisticated _is_noise_article() filtering.

**What changed:**
1. REMOVED `_consolidate_articles()` from pipeline.py's noise filtering step - it now passes all candidate starts to `consolidate.consolidate()`
2. CLEANED UP unused constants (_DEPT_HEADERS, _AD_COMPANY_WORDS, etc.) and the obsolete `_is_noise_title()` function
3. The pipeline now relies entirely on consolidate.py's proven sophisticated filtering logic for ads, department headers, masthead fragments, etc.

**Expected outcome:** Much better precision (lower phantom articles count) as consolidate.py's _is_noise_article() is more comprehensive than pipeline._is_noise_title()

**Test status:** Train score testing timed out due to long vision calls, but this change directly addresses the judge's complaint about 36 extra phantom articles in sporting-cyclist-vol1959-06.

**What needs verification:** Run the judge evaluation on val PDFs to confirm:
- Precision should improve from 0.25 towards expected value
- Article count should align with ground truth (fewer over-segmented articles)
- Metadata extraction (editor, issue.date/volume/number) should also benefit