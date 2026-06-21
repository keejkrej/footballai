import { mount } from "svelte";
import App from "./App.svelte";
import "./app.css";

const target = document.getElementById("app")!;
const app = mount(App, { target, props: {} });

export default app;
