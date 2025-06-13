import type { Metadata } from "next";
import { Inter, Poppins } from "next/font/google";
import "./globals.css";

const poppins = Poppins({
  weight: ["400", "600", "700"],
  variable: "--font-poppins",
  subsets: ["latin"],
});

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Vasavi GenZ - AI Fashion Chat",
  description: "Discover the latest fashion trends with Vasavi GenZ's AI-powered chat experience, designed for the bold and stylish.",
  icons: {
    icon: "/favicon.ico",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${poppins.variable} ${inter.variable} antialiased bg-gradient-to-br from-gray-950 to-purple-900 min-h-screen`}
      >
        {children}
      </body>
    </html>
  );
}