# QuerySelector for crawling

For each detail site `https://www.rottentomatoes.com/m/:slug`

- Description: `document.querySelector('rt-text[data-qa="synopsis-value"]').textContent`
- Title: `document.querySelector('rt-text[slot="title"]').textContent`
- Genres: `Array.from(document.querySelectorAll('rt-text[slot="metadataGenre"]')).map(node => node.textContent.replace('\/', '').trim())`
- Rating: `document.querySelector('div.category-wrap:nth-child(3) > dd:nth-child(2) > rt-text:nth-child(1)').textContent`
