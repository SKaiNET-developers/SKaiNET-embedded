'use strict'

const { execFileSync } = require('child_process')
const { writeFileSync, readFileSync, unlinkSync, mkdtempSync } = require('fs')
const { join } = require('path')
const { tmpdir } = require('os')

function findMmdc () {
  // Prefer NODE_PATH (Docker image), then local node_modules
  const paths = [
    join(process.env.NODE_PATH || '', '.bin', 'mmdc'),
    join(__dirname, '..', 'node_modules', '.bin', 'mmdc'),
  ]
  for (const p of paths) {
    try {
      require('fs').accessSync(p, require('fs').constants.X_OK)
      return p
    } catch {}
  }
  throw new Error('mmdc not found — install @mermaid-js/mermaid-cli')
}

function buildPuppeteerConfig (dir) {
  const config = { args: ['--no-sandbox', '--disable-setuid-sandbox'] }
  if (process.env.PUPPETEER_EXECUTABLE_PATH) {
    config.executablePath = process.env.PUPPETEER_EXECUTABLE_PATH
  }
  const configPath = join(dir, 'puppeteer.json')
  writeFileSync(configPath, JSON.stringify(config))
  return configPath
}

const mmdc = findMmdc()

module.exports.register = function (registry) {
  registry.block('mermaid', function () {
    this.onContext('literal')
    this.process(function (parent, reader) {
      const source = reader.getLines().join('\n')
      const dir = mkdtempSync(join(tmpdir(), 'mermaid-'))
      const input = join(dir, 'diagram.mmd')
      const output = join(dir, 'diagram.svg')
      const puppeteerConfig = buildPuppeteerConfig(dir)

      writeFileSync(input, source)
      try {
        execFileSync(mmdc, [
          '-i', input, '-o', output, '-b', 'transparent', '-p', puppeteerConfig,
        ], { stdio: 'pipe', timeout: 30000 })
        const svg = readFileSync(output, 'utf-8')
        return this.createBlock(parent, 'pass', svg)
      } catch (e) {
        console.error('mermaid render failed:', e.stderr?.toString() || e.message)
        return this.createBlock(parent, 'pass',
          `<pre class="mermaid-error">Mermaid render failed:\n${source}</pre>`)
      } finally {
        try { unlinkSync(input) } catch {}
        try { unlinkSync(output) } catch {}
        try { unlinkSync(puppeteerConfig) } catch {}
        try { require('fs').rmdirSync(dir) } catch {}
      }
    })
  })
}
