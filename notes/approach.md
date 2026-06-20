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
1. **Primary:** Integrate consolidate.py's sophisticated filtering into the main pipeline instead of using simple _is_noise_title()
2. **Secondary:** Debug why consolidate.py has better test results than pipeline.py - likely missing imports or incorrect function usage
3. **Tertiary:** Improve masthead parsing to handle edge cases where vision model misses fields but extract_masthead() should process raw text more carefully

## Dead-ends (do not retry without new information)
- Simply enhancing existing _is_noise_title() patterns without switching to consolidate.py's proven filtering logic
- Trying to fix metadata extraction in isolation without addressing the core over-segmentation issue