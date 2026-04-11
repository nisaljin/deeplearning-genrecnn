import "./globals.css";

export const metadata = {
  title: "Genre Detection Demo",
  description: "Random unseen audio demo for the CNN genre model"
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>{children}</body>
    </html>
  );
}
