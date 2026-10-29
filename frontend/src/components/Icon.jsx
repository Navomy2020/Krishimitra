const PATHS = {
  camera: "M4 8h3l1.6-2.4h6.8L17 8h3a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1zm8 2.2a3.8 3.8 0 1 0 0 7.6 3.8 3.8 0 0 0 0-7.6z",
  image: "M4 5h16a1 1 0 0 1 1 1v12a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1zm1 12h14l-4.5-6-3.5 4.5L8.5 13zM8.5 8.2a1.7 1.7 0 1 0 0 3.4 1.7 1.7 0 0 0 0-3.4z",
  back: "M11.7 5.3 5 12l6.7 6.7 1.4-1.4L8.8 13H19v-2H8.8l4.3-4.3z",
  check: "M9.6 16.2 5.4 12l-1.4 1.4 5.6 5.6L20 8.6l-1.4-1.4z",
  people: "M9 11a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7zm7 0a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM9 13c-3 0-6 1.5-6 4.2V19h12v-1.8C15 14.5 12 13 9 13zm7.2.2c2.7.3 4.8 1.6 4.8 3.8V19h-4v-1.8c0-1.4-.3-2.8-.8-4z",
  retry: "M12 5a7 7 0 1 0 6.7 9h-2.1A5 5 0 1 1 12 7c1.4 0 2.6.6 3.5 1.5L13 11h7V4l-2.6 2.6A8.9 8.9 0 0 0 12 5z",
  leaf: "M5 19C5 10 10 4.5 20 4c-.4 9.5-5.5 15-13 15-.7 0-1.4-.1-2-.3zM6.5 17.5c3-4 6.5-7 11-9.5",
};

export default function Icon({ name, size = 22 }) {
  return (
    <svg className="icon" width={size} height={size} viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      {name === "leaf" ? (
        <path d={PATHS.leaf} fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
      ) : (
        <path d={PATHS[name]} fill="currentColor" />
      )}
    </svg>
  );
}
