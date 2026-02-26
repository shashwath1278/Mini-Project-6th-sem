export const metadata = {
  title: "PlasticDeg Predictor",
  description:
    "AI prediction of plastic-degrading potential in microbial enzymes",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body style={{ margin: 0, fontFamily: "system-ui, sans-serif" }}>
        {children}
      </body>
    </html>
  );
}
