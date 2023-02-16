import { useContext } from "react";
import { MirageContext } from "../contexts/MirageContext";

function useMirage() {
  return useContext(MirageContext);
}

export default useMirage;
