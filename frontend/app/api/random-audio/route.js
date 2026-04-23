import { listBestGenreAudioFiles, listSampleAudioFiles } from "@/lib/sample-audio";

let genreRotation = [];
let genreCursor = 0;

function shuffle(values) {
  const cloned = [...values];
  for (let i = cloned.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    const tmp = cloned[i];
    cloned[i] = cloned[j];
    cloned[j] = tmp;
  }
  return cloned;
}

function getNextGenre(genres) {
  const nextSignature = genres.slice().sort().join("|");
  const currentSignature = genreRotation.slice().sort().join("|");

  if (!genreRotation.length || nextSignature !== currentSignature) {
    genreRotation = shuffle(genres);
    genreCursor = 0;
  }

  if (genreCursor >= genreRotation.length) {
    genreRotation = shuffle(genres);
    genreCursor = 0;
  }

  const genre = genreRotation[genreCursor];
  genreCursor += 1;
  return genre;
}

export async function GET() {
  try {
    const curated = listBestGenreAudioFiles();
    if (!curated.length) {
      const fallback = listSampleAudioFiles();
      if (!fallback.length) {
        return Response.json(
          { error: "No sample audio files found in frontend/public/sample-audio." },
          { status: 404 }
        );
      }

      const sample = fallback[Math.floor(Math.random() * fallback.length)];
      return Response.json(sample, { status: 200 });
    }

    const byGenre = new Map();
    for (const sample of curated) {
      const list = byGenre.get(sample.genre) || [];
      list.push(sample);
      byGenre.set(sample.genre, list);
    }

    const genres = [...byGenre.keys()];
    if (!genres.length) {
      return Response.json(
        { error: "No sample audio files found in frontend/public/sample-audio." },
        { status: 404 }
      );
    }

    const chosenGenre = getNextGenre(genres);
    const genreSamples = byGenre.get(chosenGenre) || [];
    const sample = genreSamples[Math.floor(Math.random() * genreSamples.length)];

    return Response.json(
      {
        ...sample,
        sourcePolicy: "curated-round-robin"
      },
      { status: 200 }
    );
  } catch (error) {
    return Response.json({ error: error.message || "Failed to choose random audio." }, { status: 500 });
  }
}
