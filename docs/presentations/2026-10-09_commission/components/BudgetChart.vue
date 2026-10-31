<script setup>
import results from '../results.json'
const values = [
  { label: ['0 с', 'Популяционная нормировка'], p: results.series.mean_Ws[0], color: '#8a99a5', x: 70 },
  { label: ['30 с', 'Нормировка по бюджету'], p: results.series.mean_Ws.find(x => x.seconds===30), color: '#536d82', x: 421 },
  { label: ['30 с', 'Нормировка и адаптация Wₛ'], p: results.series.ft_Ws.find(x => x.seconds===30), color: '#087f8c', x: 772 },
]
const base = 272, scale = 3.25
</script>
<template>
  <svg viewBox="0 0 1130 363" style="width:100%; height:326px" role="img" aria-label="Разделение изменения нормировки и дообучения W_s при 30 секундах">
    <g font-family="Arial, sans-serif">
      <text x="0" y="22" style="font-size:19px" fill="#5a6976">Top-10, %</text>
      <line x1="30" x2="1100" :y1="base" :y2="base" stroke="#ccd6de" stroke-width="2" />
      <g v-for="v in values" :key="v.x">
        <rect :x="v.x" :y="base-v.p.top10*scale" width="248" :height="v.p.top10*scale" :fill="v.color" />
        <line :x1="v.x+124" :x2="v.x+124" :y1="base-(v.p.top10+v.p.ci95)*scale" :y2="base-(v.p.top10-v.p.ci95)*scale" stroke="#152738" stroke-width="2" />
        <line v-for="sign in [-1,1]" :key="sign" :x1="v.x+113" :x2="v.x+135" :y1="base-(v.p.top10+sign*v.p.ci95)*scale" :y2="base-(v.p.top10+sign*v.p.ci95)*scale" stroke="#152738" stroke-width="2" />
        <text :x="v.x+124" :y="base-(v.p.top10+v.p.ci95)*scale-15" text-anchor="middle" style="font-size:39px" font-weight="700" :fill="v.color">{{ v.p.top10.toFixed(1) }}%</text>
        <text :x="v.x+124" y="305" text-anchor="middle" style="font-size:24px" font-weight="600" fill="#152738">{{ v.label[0] }}</text>
        <text :x="v.x+124" y="338" text-anchor="middle" style="font-size:20px" fill="#5a6976">{{ v.label[1] }}</text>
      </g>
    </g>
  </svg>
</template>
