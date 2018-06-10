# word-mesh
A wordcloud/wordmesh generator that allows users to extract keywords from text, and create a simple and interpretable wordcloud.


## Why word-mesh?

Most popular open-source wordcloud generators ([word_cloud](https://github.com/amueller/word_cloud), [d3-cloud](https://github.com/jasondavies/d3-cloud), [echarts-wordcloud](https://github.com/ecomfe/echarts-wordcloud)) focus more on the aesthetics of the visualization, than on conveying meaningful information. **word-mesh** strikes a balance between the two and distinguishes itself by offering the following features:

 - *intelligent keyword extraction*: In addition to 'word frequency' based extraction techniques, word-mesh supports graph based methods like **textrank**, **sgrank**, etc.
 - *word clustering*: Words can be grouped together on the canvas based on the following criteria - semantic meaning, co-occurence, frequency, custom criteria. This allows for some [interesting visualizations](#examples).
 - *pos /entity type filtering*: Extracted keywords can be filtered based on pos tags and named entity types.
 - *fontcolors and fontsizes*: These can be set based on the following criteria - word frequency, pos-tags, word-longevity.

## Examples

![Sample](examples/wordmesh_risk.png?raw=true = 300x500)

