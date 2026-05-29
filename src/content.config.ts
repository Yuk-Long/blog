import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const blogCollection = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/blog' }),
  schema: z.object({
    title: z.string(),
    date: z.date(),
    description: z.string().default(''),
    tags: z.array(z.string()).default([]),
    readingTime: z.number().default(1),
    author: z.string().default(''),
  }),
});

export const collections = {
  blog: blogCollection,
};