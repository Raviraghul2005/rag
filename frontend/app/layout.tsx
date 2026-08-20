import type { Metadata } from "next";
import { Anton, Archivo, Inter, Kaushan_Script, Lora } from "next/font/google";
import "./globals.css";

const anton = Anton({ variable: "--font-anton", weight: "400", subsets: ["latin"] });
const archivo = Archivo({ variable: "--font-archivo", subsets: ["latin"] });
const inter = Inter({ variable: "--font-inter", subsets: ["latin"] });
const lora = Lora({ variable: "--font-lora", subsets: ["latin"], style: ["normal", "italic"] });
const kaushan = Kaushan_Script({ variable: "--font-kaushan", weight: "400", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "RAGInGoa — Voice RAG in 200ms",
  description:
    "Speak a question in an Indian language, get a grounded, cited answer — or an explicit refusal — with a live per-stage latency breakdown.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${anton.variable} ${archivo.variable} ${inter.variable} ${lora.variable} ${kaushan.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
