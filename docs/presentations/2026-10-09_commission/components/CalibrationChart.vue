<script setup>
import results from '../results.json'
const left = 72, right = 1065, top = 51, bottom = 354
const x = s => left + s / 882 * (right - left)
const y = p => bottom - p / 100 * (bottom - top)
const series = [
  { data: results.series.ft_Ws, color: '#087f8c', label: 'Дообучение Wₛ', labelX: 72 },
  { data: results.series.ft_all, color: '#c67631', label: 'Дообучение всей сети', labelX: 376 },
]
const line = data => data.map((p, i) => `${i ? 'L' : 'M'}${x(p.seconds)},${y(p.top10)}`).join(' ')
const band = data => `${data.map((p, i) => `${i ? 'L' : 'M'}${x(p.seconds)},${y(p.top10 + p.ci95)}`).join(' ')} ${[...data].reverse().map(p => `L${x(p.seconds)},${y(p.top10 - p.ci95)}`).join(' ')} Z`
</script>
<template>
  <svg viewBox="0 0 1130 455" style="width:100%; height:410px" role="img" aria-label="Кривая качества калибровки W_s и всей сети с 95-процентными интервалами">
    <g font-family="Arial, sans-serif">
      <text x="0" y="22" style="font-size:19px" fill="#5a6976">Top-10, %</text>
      <g v-for="tick in [0,20,40,60,80,100]" :key="tick">
        <line :x1="left" :x2="right" :y1="y(tick)" :y2="y(tick)" stroke="#dde4e9" stroke-width="1" />
        <text :x="left-14" :y="y(tick)+6" text-anchor="end" style="font-size:20px" fill="#5a6976">{{ tick }}</text>
      </g>
      <line :x1="left" :x2="right" :y1="y(results.chance_top10)" :y2="y(results.chance_top10)" stroke="#a2adb6" stroke-width="1.5" stroke-dasharray="4 6" />
      <text :x="right-8" :y="y(results.chance_top10)-8" text-anchor="end" style="font-size:19px" fill="#788590">Случай: 10.1%</text>
      <line :x1="left" :x2="right" :y1="y(results.joint_top10)" :y2="y(results.joint_top10)" stroke="#536373" stroke-width="2" stroke-dasharray="9 6" />
      <text :x="right-8" :y="y(results.joint_top10)+23" text-anchor="end" style="font-size:19px" fill="#536373">Joint: 79.5%</text>
      <g v-for="s in series" :key="s.label">
        <path :d="band(s.data)" :fill="s.color" fill-opacity="0.10" />
        <path :d="line(s.data)" :stroke="s.color" stroke-width="4" fill="none" />
        <circle v-for="p in s.data" :key="p.seconds" :cx="x(p.seconds)" :cy="y(p.top10)" r="5" :fill="s.color">
          <title>{{ s.label }}: {{ p.seconds }} с, {{ p.top10.toFixed(1) }}%, 95% ДИ ±{{ p.ci95.toFixed(1) }} п.п.</title>
        </circle>
      </g>
      <circle :cx="x(0)" :cy="y(results.series.mean_Ws[0].top10)" r="6" fill="#152738" />
      <text :x="x(0)+12" :y="y(results.series.mean_Ws[0].top10)+25" style="font-size:19px" fill="#152738">Без данных: 41.8%</text>
      <g v-for="tick in [0,120,300,600,882]" :key="tick">
        <text :x="x(tick)" :y="bottom+28" text-anchor="middle" style="font-size:20px" fill="#5a6976">{{ tick === 882 ? '14.7' : tick/60 }}</text>
      </g>
      <text :x="(left+right)/2" :y="bottom+62" text-anchor="middle" style="font-size:21px" fill="#5a6976">Бюджет калибровки, мин</text>
      <g v-for="s in series" :key="s.label+'legend'">
        <line :x1="s.labelX" :x2="s.labelX+32" y1="443" y2="443" :stroke="s.color" stroke-width="4" />
        <text :x="s.labelX+43" y="450" style="font-size:21px" fill="#152738">{{ s.label }}</text>
      </g>
      <text x="838" y="450" style="font-size:19px" fill="#5a6976">Заливка: 95% ДИ</text>
    </g>
  </svg>
</template>
