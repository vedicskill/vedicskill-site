import { defineUserConfig } from 'vuepress'
import { defaultTheme } from '@vuepress/theme-default'
import { webpackBundler } from '@vuepress/bundler-webpack'

export default defineUserConfig({
  base: '/vedicskill-site/',
  lang: 'en-US',
  title: 'Vedicskill',
  description: 'Data-driven analytics and learning with Vedicskill',
  head: [
    ['meta', { name: 'viewport', content: 'width=device-width, initial-scale=1' }],
    ['link', { rel: 'icon', href: '/images/favicon_io/favicon.ico' }],
  ],

  theme: defaultTheme({
//    logo: '/logo.png',
    navbar: [
      { text: 'Home', link: '/' },
      { text: 'Courses', link: 'https://www.udemy.com/user/freeai-space/' },
      {
        text: 'Documentation',
        children: [
          { text: 'MongoDB Atlas Vector DB', link: '/mongodb/mongodb.html' },
          { text: 'Statistics for Data Science', link: '/statistics/statistics.html' },
        ],
      },
      { text: 'About Us', link: '/about.html' },
    ],
    sidebar: {
    '/': [
        {
          text: 'Getting Started',
          children: [
            '/',                      // points to docs/README.md (homepage)
            '/about.md',
          ],
        },
      ],
      '/mongodb/': [
        {
          text: 'MongoDB Docs',
          collapsible: true,
          children: [
            '/mongodb/README.md',
            '/mongodb/mongodb.md',
          ],
        },
      ],
      '/statistics/': [
        {
          text: 'Statistics Docs',
          collapsible: true,
          children: [
            '/statistics/README.md',
            '/statistics/statistics.md',
          ],
        },
      ],
    },
    editLink: false,
    contributors: false,
    lastUpdated: false,
  }),

  bundler: webpackBundler({
    // keep defaults; add options here if needed
  }),
})
