import { spawnSync } from 'node:child_process'
import { readFile, writeFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { PDFDocument } from 'pdf-lib'

const cwd = fileURLToPath(new URL('..', import.meta.url))
const result = spawnSync('npm', ['run', 'export', '--', ...process.argv.slice(2)], {
  cwd,
  stdio: 'inherit',
})
if (result.error) throw result.error
if (result.status !== 0) process.exit(result.status ?? 1)

// Slidev 53 PDF export can ignore --range; copy the ten main pages explicitly.
const full = await PDFDocument.load(await readFile(new URL('../commission.pdf', import.meta.url)))
if (full.getPageCount() !== 14) throw new Error('Expected 10 main and 4 appendix slides')
const main = await PDFDocument.create()
for (const page of await main.copyPages(full, Array.from({ length: 10 }, (_, i) => i))) {
  main.addPage(page)
}
main.setTitle('Перенос МЭГ-декодера речи на нового участника — основной доклад')
main.setAuthor('Михаил')
await writeFile(new URL('../commission-main.pdf', import.meta.url), await main.save())
console.log('Exported commission-main.pdf: 10 main slides')
